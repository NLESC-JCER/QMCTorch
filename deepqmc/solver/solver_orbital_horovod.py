import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd

from deepqmc.solver.solver_base import SolverBase
from deepqmc.solver.torch_utils import (DataSet, Loss,
                                        ZeroOneClipper, OrthoReg)


def printd(rank, *args):
    if rank == 1:
        print(*args)


class SolverOrbital(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None,
                 scheduler=None):

        SolverBase.__init__(self, wf, sampler, optimizer)
        self.scheduler = scheduler

        # task
        self.configure(task='geo_opt')

        # esampling
        self.resampling(ntherm=-1, resample=100,
                        resample_from_last=True,
                        resample_every=1)

        # observalbe
        self.observable(['local_energy'])

        # distributed model
        self.save_model = 'model.pth'

        if self.wf.cuda:
            self.device = torch.device('cuda')
            self.sampler.cuda = True
            self.sampler.walkers.cuda = True
        else:
            self.device = torch.device('cpu')

        if 'lpos_needed' not in self.opt.__dict__.keys():
            self.opt.lpos_needed = False

        hvd.init()
        hvd.broadcast_optimizer_state(self.opt, root_rank=0)
        self.opt = hvd.DistributedOptimizer(self.opt,
                                            named_parameters=self.wf.named_parameters())

    def configure(self, task='wf_opt', freeze=None):
        '''Configure the optimzier for specific tasks.'''
        self.task = task

        if task == 'geo_opt':
            self.wf.ao.atom_coords.requires_grad = True
            self.wf.ao.bas_exp.requires_grad = False
            for param in self.wf.mo.parameters():
                param.requires_grad = False
            self.wf.fc.weight.requires_grad = False

        elif task == 'wf_opt':
            self.wf.ao.bas_exp.requires_grad = True
            for param in self.wf.mo.parameters():
                param.requires_grad = True
            self.wf.fc.weight.requires_grad = True
            self.wf.ao.atom_coords.requires_grad = False

            if freeze is not None:
                if not isinstance(freeze, list):
                    freeze = [freeze]
                for name in freeze:
                    if name.lower() == 'ci':
                        self.wf.fc.weight.requires_grad = False
                    elif name.lower() == 'mo':
                        for param in self.wf.mo.parameters():
                            param.requires_grad = False
                    elif name.lower() == 'bas_exp':
                        self.wf.ao.bas_exp.requires_grad = False
                    else:
                        opt_freeze = ['ci', 'mo', 'bas_exp']
                        raise ValueError(
                            'Valid arguments for freeze are :', opt_freeze)

    def run(self, nepoch, batchsize=None, loss='variance', num_threads=1):
        '''Train the model.

        Arg:
            nepoch : number of epoch
            batchsize : size of the minibatch, if None take all points at once
            loss : loss used ('energy','variance' or callable (for supervised)
        '''

        hvd.broadcast_parameters(self.wf.state_dict(), root_rank=0)
        torch.set_num_threads(num_threads)

        # sample the wave function
        pos = self.sample(ntherm=self.resample.ntherm)

        # handle the batch size
        if batchsize is None:
            batchsize = len(pos)

        # change the number of steps
        _nstep_save = self.sampler.nstep
        self.sampler.nstep = self.resample.resample

        # create the data loader
        self.dataset = DataSet(pos)
        self.sampler = DistributedSampler(self.dataset,
                                          num_replicas=hvd.size(),
                                          rank=hvd.rank())
        if self.cuda:
            kwargs = {'num_workers': num_threads, 'pin_memory': True}
        else:
            kwargs = {'num_workers': num_threads}

        self.dataloader = DataLoader(self.dataset,
                                     batch_size=batchsize,
                                     sampler=self.sampler,
                                     **kwargs)

        # get the loss
        self.loss = Loss(self.wf, method=loss)

        # orthogonalization penalty for the MO coeffs
        self.ortho_loss = OrthoReg()

        # clipper for the fc weights
        clipper = ZeroOneClipper()

        cumulative_loss = []
        min_loss = 1E3

        # get the initial observalbe

        self.get_observable(self.obs_dict, pos)
        for n in range(nepoch):
            printd(hvd.rank(), '----------------------------------------')
            printd(hvd.rank(), 'epoch %d' % n)

            cumulative_loss = 0
            for data in self.dataloader:

                lpos = data.to(self.device)

                loss, eloc = self.loss(lpos)
                if self.wf.mo.weight.requires_grad:
                    loss += self.ortho_loss(self.wf.mo.weight)
                cumulative_loss += loss

                # compute gradients
                self.opt.zero_grad()
                loss.backward()

                # optimize
                if self.opt.lpos_needed:
                    self.opt.step(lpos)
                else:
                    self.opt.step()

                if self.wf.fc.clip:
                    self.wf.fc.apply(clipper)

            cumulative_loss = self.metric_average(cumulative_loss, 'cum_loss')
            if hvd.rank() == 0:
                if cumulative_loss < min_loss:
                    min_loss = self.save_checkpoint(
                        n, cumulative_loss, self.save_model)

            self.get_observable(self.obs_dict, pos, local_energy=eloc)
            self.print_observable(cumulative_loss)

            printd(hvd.rank(), '----------------------------------------')

            # resample the data
            if (n % self.resample.resample_every == 0) or (n == nepoch-1):
                if self.resample.resample_from_last:
                    pos = pos.clone().detach().to(self.device)
                else:
                    pos = None
                pos = self.sample(
                    pos=pos, ntherm=self.resample.ntherm, with_tqdm=False)

                self.dataloader.dataset.data = pos

            if self.task == 'geo_opt':
                self.wf.update_mo_coeffs()

            if self.scheduler is not None:
                self.scheduler.step()

        # restore the sampler number of step
        self.sampler.nstep = _nstep_save

    @staticmethod
    def metric_average(val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def save_traj(self, fname):

        f = open(fname, 'w')
        xyz = self.obs_dict['geometry']
        natom = len(xyz[0])
        nm2bohr = 1.88973
        for snap in xyz:
            f.write('%d \n\n' % natom)
            for at in snap:
                f.write('%s % 7.5f % 7.5f %7.5f\n' % (at[0], at[1][0]/nm2bohr,
                                                      at[1][1]/nm2bohr,
                                                      at[1][2]/nm2bohr))
            f.write('\n')
        f.close()
