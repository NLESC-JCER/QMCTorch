import torch
from torch.utils.data import DataLoader
import warnings

from .solver_base import SolverBase
from qmctorch.utils import (DataSet, Loss, OrthoReg)

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    pass


def printd(rank, *args):
    if rank == 0:
        print(*args)


class SolverOrbitalHorovod(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None, scheduler=None):
        """Horovod distributed solver

        Keyword Arguments:
            wf {WaveFunction} -- WaveFuntion object (default: {None})
            sampler {SamplerBase} -- Samppler (default: {None})
            optimizer {torch.optim} -- Optimizer (default: {None})
            scheduler (torch.schedul) -- Scheduler (default: {None})
        """

        SolverBase.__init__(self, wf, sampler, optimizer)

        hvd.broadcast_optimizer_state(self.opt, root_rank=0)
        self.opt = hvd.DistributedOptimizer(
            self.opt, named_parameters=self.wf.named_parameters())

        self.sampler.nwalkers //= hvd.size()
        self.sampler.walkers.nwalkers //= hvd.size()

    def run(self, nepoch, batchsize=None, loss='variance', num_threads=1):
        '''Train the model.

        Arg:
            nepoch : number of epoch
            batchsize : size of the minibatch, if None take all points at once
            loss : loss used ('energy','variance' or callable (for supervised)
        '''

        if 'lpos_needed' not in self.opt.defaults:
            self.opt.defaults['lpos_needed'] = False

        self.wf.train()

        hvd.broadcast_parameters(self.wf.state_dict(), root_rank=0)
        torch.set_num_threads(num_threads)

        # sample the wave function
        pos = self.sampler(ntherm=self.resample.ntherm)
        pos.requires_grad_(False)

        # handle the batch size
        if batchsize is None:
            batchsize = len(pos)

        # change the number of steps
        _nstep_save = self.sampler.nstep
        self.sampler.nstep = self.resample.resample

        # create the data loader
        self.dataset = DataSet(pos)

        if self.cuda:
            kwargs = {'num_workers': num_threads, 'pin_memory': True}
        else:
            kwargs = {'num_workers': num_threads}

        self.dataloader = DataLoader(self.dataset,
                                     batch_size=batchsize,
                                     **kwargs)

        # get the loss
        self.loss = Loss(self.wf, method=loss)

        # orthogonalization penalty for the MO coeffs
        self.ortho_loss = OrthoReg()

        cumulative_loss = []
        min_loss = 1E3

        # get the initial observalbe

        self.get_observable(self.obs_dict, pos)
        for n in range(nepoch):

            printd(
                hvd.rank(),
                '----------------------------------------')
            printd(hvd.rank(), 'epoch %d' % n)

            cumulative_loss = 0.
            for data in self.dataloader:

                lpos = data.to(self.device)
                lpos.requires_grad = True

                loss, eloc = self.loss(lpos)
                if self.wf.mo.weight.requires_grad:
                    loss += self.ortho_loss(self.wf.mo.weight)
                cumulative_loss += loss

                # compute gradients
                self.opt.zero_grad()
                loss.backward()

                # optimize
                if 'lpos_needed' in self.opt.defaults:
                    self.opt.step(lpos)
                else:
                    self.opt.step()

            cumulative_loss = self.metric_average(
                cumulative_loss, 'cum_loss')
            if hvd.rank() == 0:
                if cumulative_loss < min_loss:
                    min_loss = self.save_checkpoint(
                        n, cumulative_loss, self.save_model)

            self.get_observable(self.obs_dict, pos, local_energy=eloc)
            self.print_observable(cumulative_loss)

            printd(
                hvd.rank(),
                '----------------------------------------')

            # resample the data
            if (n % self.resample.resample_every == 0) or (
                    n == nepoch - 1):
                if self.resample.resample_from_last:
                    pos = pos.clone().detach().to(self.device)
                else:
                    pos = None

                pos = self.sampler(pos=pos,
                                   ntherm=self.resample.ntherm,
                                   with_tqdm=False)
                pos.requires_grad_(False)

                self.dataloader.dataset.data = pos

            if self.task == 'geo_opt':
                self.wf.update_mo_coeffs()

            if self.scheduler is not None:
                self.scheduler.step()

        # restore the sampler number of step
        self.sampler.nstep = _nstep_save

    def single_point(self, pos=None, prt=True, ntherm=-1, ndecor=100):
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
        pos = self.sampler(ntherm=self.resample.ntherm)
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
        e = torch.mean(eloc_all)
        s = torch.var(eloc_all)

        if prt:
            printd(hvd.rank(), 'Energy   : ', e)
            printd(hvd.rank(), 'Variance : ', s)
        return pos, e, s

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
