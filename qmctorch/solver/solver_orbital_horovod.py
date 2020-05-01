import torch
from torch.utils.data import DataLoader
import warnings

from .solver_orbital import SolverOrbital
from qmctorch.utils import (DataSet, Loss, OrthoReg)

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    pass


def printd(rank, *args):
    if rank == 0:
        print(*args)


class SolverOrbitalHorovod(SolverOrbital):

    def __init__(self, wf=None, sampler=None, optimizer=None, scheduler=None, output=None):
        """Horovod distributed solver

        Keyword Arguments:
            wf {WaveFunction} -- WaveFuntion object (default: {None})
            sampler {SamplerBase} -- Samppler (default: {None})
            optimizer {torch.optim} -- Optimizer (default: {None})
            scheduler (torch.schedul) -- Scheduler (default: {None})
        """

        SolverBase.__init__(self, wf, sampler,
                            optimizer, scheduler, output)

        hvd.broadcast_optimizer_state(self.opt, root_rank=0)
        self.opt = hvd.DistributedOptimizer(
            self.opt, named_parameters=self.wf.named_parameters())

        self.sampler.nwalkers //= hvd.size()
        self.sampler.walkers.nwalkers //= hvd.size()

    def run(self, nepoch, batchsize=None, loss='energy',
            clip_loss=False, grad='manual', hdf5_group=None,
            num_threads=1):
        """Run the optimization

        Args:
            nepoch (int): Number of optimziation step
            batchsize (int, optional): Number of sample in a mini batch.
                                       If None, all samples are used.
                                       Defaults to None.
            loss (str, optional): merhod to compute the loss: variance or energy.
                                  Defaults to 'energy'.
            clip_loss (bool, optional): Clip the loss values at +/- 5std.
                                        Defaults to False.
            grad (str, optional): method to compute the gradients: 'auto' or 'manual'.
                                  Defaults to 'auto'.
            hdf5_group (str, optional): name of the hdf5 group where to store the data.
                                        Defaults to wf.task.
        """

        # observalbe
        if not hasattr(self, 'observable'):
            self.track_observable(['local_energy'])

        self.evaluate_gradient = {
            'auto': self.evaluate_grad_auto,
            'manual': self.evaluate_grad_manual}[grad]

        if 'lpos_needed' not in self.opt.defaults:
            self.opt.defaults['lpos_needed'] = False

        self.wf.train()

        hvd.broadcast_parameters(self.wf.state_dict(), root_rank=0)
        torch.set_num_threads(num_threads)

        # sample the wave function
        pos = self.sampler(self.wf.pdf)
        # pos.requires_grad_(False)

        # get the loss
        self.loss = Loss(self.wf, method=loss, clip=clip_loss)
        self.loss.use_weight = (
            self.resampling_options.resample_every > 1)

        # orthogonalization penalty for the MO coeffs
        self.ortho_loss = OrthoReg()

        # handle the batch size
        if batchsize is None:
            batchsize = len(pos)

        # change the number of steps/walker size
        _nstep_save = self.sampler.nstep
        _ntherm_save = self.sampler.ntherm
        _nwalker_save = self.sampler.walkers.nwalkers
        if self.resampling_options.mode == 'update':
            self.sampler.ntherm = -1
            self.sampler.nstep = self.resampling_options.nstep_update
            self.sampler.walkers.nwalkers = pos.shape[0]
            self.sampler.nwalkers = pos.shape[0]

        # create the data loader
        self.dataset = DataSet(pos)

        if self.cuda:
            kwargs = {'num_workers': num_threads, 'pin_memory': True}
        else:
            kwargs = {'num_workers': num_threads}

        self.dataloader = DataLoader(self.dataset,
                                     batch_size=batchsize,
                                     **kwargs)
        cumulative_loss = []
        min_loss = 1E3

        # get the initial observalbe
        self.store_observable(pos)

        for n in range(nepoch):

            printd(
                hvd.rank(),
                '----------------------------------------')
            printd(hvd.rank(), 'epoch %d' % n)

            cumulative_loss = 0.
            for ibatch, data in enumerate(self.dataloader):

                # get data
                lpos = data.to(self.device)
                lpos.requires_grad = True

                # get the gradient
                loss, eloc = self.evaluate_gradient(lpos)
                cumulative_loss += loss

                # optimize the parameters
                self.optimization_step(lpos)

                # observable
                self.store_observable(
                    pos, local_energy=eloc, ibatch=ibatch)

            cumulative_loss = self.metric_average(cumulative_loss,
                                                  'cum_loss')

            if hvd.rank() == 0:
                if cumulative_loss < min_loss:
                    min_loss = self.save_checkpoint(
                        n, cumulative_loss, self.save_model)

            self.print_observable(cumulative_loss)

            printd(hvd.rank(),
                   '----------------------------------------')

            # resample the data
            pos = self.resample(n, pos)

            if self.task == 'geo_opt':
                self.wf.update_mo_coeffs()

            if self.scheduler is not None:
                self.scheduler.step()

        # restore the sampler number of step
        self.sampler.nstep = _nstep_save
        self.sampler.ntherm = _ntherm_save
        self.sampler.walkers.nwalkers = _nwalker_save
        self.sampler.nwalkers = _nwalker_save

        # dump
        if hdf5_group is None:
            hdf5_group = self.task
        dump_to_hdf5(self.observable, self.hdf5file, hdf5_group)
        add_group_attr(self.hdf5file, hdf5_group, {'type': 'opt'})

        return self.observable

    def single_point(self, pos=None, prt=True):
        """Performs a single point calculation

        Keyword Arguments:
            pos {torch.tensor} -- positions of the walkers If none, sample
                                  (default: {None})
            prt {bool} -- print energy/variance values (default: {True})
            ntherm {int} -- number of MC steps to thermalize (default: {-1})
            ndecor {int} -- number of MC step to decorelate  (default: {100})

        Returns:
            tuple -- (position, energy, variance)
        """

        self.wf.eval()
        num_threads = 1
        hvd.broadcast_parameters(self.wf.state_dict(), root_rank=0)
        torch.set_num_threads(num_threads)

        # sample the wave function
        pos = self.sampler(self.wf.pdf)
        pos.requires_grad_(False)

        # handle the batch size
        batchsize = len(pos)

        # create the data loader
        self.dataset = DataSet(pos)

        if self.cuda:
            kwargs = {'num_workers': num_threads, 'pin_memory': True}
        else:
            kwargs = {'num_workers': num_threads}
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=batchsize,
                                     **kwargs)

        for data in self.dataloader:

            lpos = data.to(self.device)
            lpos.requires_grad = True
            eloc = self.wf.local_energy(lpos)

        eloc_all = hvd.allgather(eloc, name='local_energies')
        e, s = torch.mean(eloc_all), torch.var(eloc_all)
        err = self.wf.sampling_error(eloc_all)

        # print data
        print('Energy   : ', e.detach().item(),
              ' +/- ', err.detach().item())
        print('Variance : ', s.detach().item())

        # dump data to hdf5
        obs = SimpleNamespace(
            pos=pos,
            local_energy=el,
            energy=e,
            variance=s,
            error=err
        )

        # dump to file
        if hvd.rank() == 0:

            dump_to_hdf5(obs,
                         self.hdf5file,
                         root_name=hdf5_group)
            add_group_attr(self.hdf5file, hdf5_group,
                           {'type': 'single_point'})

        return obs

    @staticmethod
    def metric_average(val, name):
        """Average a give quantity over all processes

        Arguments:
            val {torch.tensor} -- data to average
            name {str} -- name of the data

        Returns:
            torch.tensor -- Averaged quantity
        """
        tensor = val.clone().detach()
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()
