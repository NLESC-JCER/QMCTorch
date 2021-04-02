
import torch
from .elec_elec.pade_jastrow import PadeJastrow
from .elec_elec.pade_jastrow_polynomial import PadeJastrowPolynomial
from .elec_elec.electron_electron_generic import ElectronElectronGeneric
from ... import log


def set_jastrow(jastrow_type, nup, ndown, cuda, **kwargs):
    """Set the jastrow calculator

    Args:
        jastrow_type (str): name of the jastrow
        nup (int): number of up electrons
        ndown (int): number of down electrons
        cuda (bool): use cuda
    """

    if isinstance(jastrow_type, str):
        if jastrow_type == 'pade_jastrow':
            return PadeJastrow(nup, ndown, w=1., cuda=cuda)

        elif jastrow_type.startswith('pade_jastrow('):
            order = int(jastrow_type.split('(')[1][0])
            return PadeJastrowPolynomial(nup, ndown, order, cuda=cuda)

        elif jastrow_type == 'scaled_pade_jastrow':
            return PadeJastrow(nup, ndown, w=1., scale=True, scale_factor=0.6, cuda=cuda)

    # load generic jastrow without args
    elif issubclass(jastrow_type, torch.nn.Module):
        return ElectronElectronGeneric(nup, ndown, jastrow_type, cuda, **kwargs)

    else:
        valid_names = ['pade_jastrow',
                       'pade_jastrow_(n)',
                       'scaled_pade_jastrow']
        log.info(
            '   Error : Jastrow form not recognized. Options are :')
        for n in valid_names:
            log.info('         : {0}', n)
        log.info(
            '         : if generic jastrow it should be subclass nn.Module')
        raise ValueError('Jastrow type not supported')
