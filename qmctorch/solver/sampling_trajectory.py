from types import SimpleNamespace
from copy import deepcopy
from tqdm import tqdm
import numpy as np

from .. import log
from ..utils import dump_to_hdf5, add_group_attr


def SamplingTrajectory(wf, sampler, with_tqdm=True, hdf5_group='sampling_trajectory'):
    """Compute the local energy along a sampling trajectory

    Args:
        pos (torch.tensor): positions of the walkers along the trajectory
        hdf5_group (str, optional): name of the group where to store the data.
                                    Defaults to 'sampling_trajecory'
    Returns:
        SimpleNamespace : contains energy/positions/
    """
    log.info('')
    log.info('  Sampling trajectory')

    log.info(
        '  WaveFunction        : {0}', self.wf.__class__.__name__)
    for x in self.wf.__repr__().split('\n'):
        log.debug('   ' + x)

    log.info(
        '  Sampler             : {0}', self.sampler.__class__.__name__)
    for x in self.sampler.__repr__().split('\n'):
        log.debug('   ' + x)

    # make a copy of the sampler
    traj_sampler = deepcopy(sampler)
    traj_sampler.ntherm = 0
    traj_sampler.nstep = sampler.ntherm
    traj_sampler.ndecor = 1
    traj_sampler.nsample = traj_sampler.nwalkers * traj_sampler.nstep

    # sample
    pos = traj_sampler(wf.pdf, with_tqdm=with_tqdm)

    ndim = pos.shape[-1]
    p = pos.view(-1, sampler.nwalkers, ndim)
    el = []
    rng = tqdm(p, desc='INFO:QMCTorch|  Energy  ',
               disable=not with_tqdm)
    for ip in rng:
        el.append(wf.local_energy(ip).cpu().detach().numpy())

    el = np.array(el).squeeze(-1)
    obs = SimpleNamespace(local_energy=np.array(el), pos=pos)

    if hdf5file is not None:

        hdf5_group = 'sampling_trajectory'
        dump_to_hdf5(obs,
                     hdf5file, hdf5_group)

        add_group_attr(hdf5file, hdf5_group,
                       {'type': 'sampling_traj'})
    return obs
