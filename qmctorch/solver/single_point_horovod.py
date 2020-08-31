import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader
import warnings
from time import time
from .solver_orbital import SolverOrbital
from qmctorch.utils import (
    DataSet, Loss, OrthoReg, dump_to_hdf5, add_group_attr)
from .. import logd

from mpi4py import MPI

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    pass


def SinglePointHorovod(wf, sampler, with_tqdm=True, hdf5file=None):
    """Performs a single point calculation

    Args:
        with_tqdm (bool, optional): use tqdm for samplig. Defaults to True.
        hdf5_group (str, optional): hdf5 group where to store the data.
                                    Defaults to 'single_point'.

    Returns:
        SimpleNamespace: contains the local energy, positions, ...
    """

    logd(hvd.rank(), '')
    logd(hvd.rank(), '  Single Point Calculation : {nw} walkers | {ns} steps | {nproc}'.format(
        nw=self.sampler.nwalkers, ns=self.sampler.nstep, nproc=hvd.size()))

    # reduce the sampling size of each process
    sampler.nwalkers //= hvd.size()
    sampler.walkers.nwalkers //= hvd.size()

    # check if we have to compute and store the grads
    grad_mode = torch.no_grad()
    if self.wf.kinetic == 'auto':
        grad_mode = torch.enable_grad()

    # distribute the calculation
    num_threads = 1
    hvd.broadcast_parameters(self.wf.state_dict(), root_rank=0)
    torch.set_num_threads(num_threads)

    with grad_mode:

        # sample the wave function
        pos = self.sampler(self.wf.pdf)
        if self.wf.cuda and pos.device.type == 'cpu':
            pos = pos.to(self.device)

        # compute energy/variance/error
        eloc = self.wf.local_energy(pos)
        e, s, err = torch.mean(eloc), torch.var(
            eloc), self.wf.sampling_error(eloc)

        # gather all data
        eloc_all = hvd.allgather(eloc, name='local_energies')
        e, s, err = torch.mean(eloc_all), torch.var(
            eloc_all), self.wf.sampling_error(eloc_all)

        # print
        if hvd.rank() == 0:
            log.options(style='percent').info(
                '  Energy   : %f +/- %f' % (e.detach().item(), err.detach().item()))
            log.options(style='percent').info(
                '  Variance : %f' % s.detach().item())

        # dump data to hdf5
        obs = SimpleNamespace(
            pos=pos,
            local_energy=eloc_all,
            energy=e,
            variance=s,
            error=err
        )

        # dump to file
        if hdf5file is not None:
            if hvd.rank() == 0:
                hdf5_group = 'single_point'
                dump_to_hdf5(obs,
                             hdf5file,
                             root_name=hdf5_group)
                add_group_attr(hdf5file, hdf5_group,
                               {'type': 'single_point'})

    return obs
