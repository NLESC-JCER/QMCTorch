from types import SimpleNamespace
import numpy as np
import torch

from .. import log, logd
from ..utils import dump_to_hdf5, add_group_attr


def SinglePoint(wf, sampler, with_tqdm=True, hdf5file=None):
    """Performs a single point calculation

    Args:
        with_tqdm (bool, optional): use tqdm for samplig. Defaults to True.
        hdf5_group (str, optional): hdf5 group where to store the data.
                                    Defaults to 'single_point'.

    Returns:
        SimpleNamespace: contains the local energy, positions, ...
    """

    log.info(
        '  WaveFunction        : {0}', wf.__class__.__name__)
    for x in wf.__repr__().split('\n'):
        log.debug('   ' + x)

    log.info(
        '  Sampler             : {0}', sampler.__class__.__name__)
    for x in sampler.__repr__().split('\n'):
        log.debug('   ' + x)

    log.info('')
    log.info('  Single Point Calculation : {nw} walkers | {ns} steps'.format(
        nw=sampler.nwalkers, ns=sampler.ntherm))

    # check if we have to compute and store the grads
    grad_mode = torch.no_grad()
    if wf.kinetic == 'auto':
        grad_mode = torch.enable_grad()

    with grad_mode:

        #  get the position and put to gpu if necessary
        pos = sampler(wf.pdf, with_tqdm=with_tqdm)
        if wf.cuda and pos.device.type == 'cpu':
            pos = pos.to(wf.device)

        # compute energy/variance/error
        el = wf.local_energy(pos)
        e, s, err = torch.mean(el), torch.var(
            el), wf.sampling_error(el)

        # print data
        log.options(style='percent').info(
            '  Energy   : %f +/- %f' % (e.detach().item(), err.detach().item()))
        log.options(style='percent').info(
            '  Variance : %f' % s.detach().item())

        # dump data to hdf5
        obs = SimpleNamespace(
            pos=pos,
            local_energy=el,
            energy=e,
            variance=s,
            error=err
        )
        if hdf5file is not None:
            hdf5_group = 'single_point'
            dump_to_hdf5(obs,
                         hdf5file,
                         root_name=hdf5_group)
            add_group_attr(hdf5file, hdf5_group,
                           {'type': 'single_point'})

    return obs
