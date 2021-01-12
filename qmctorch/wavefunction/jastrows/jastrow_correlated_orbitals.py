import torch
from .pade_jastrow_orbital import PadeJastrowOrbital
from .unity_jastrow_orbital import UnityJastrowOrbital
from .generic_jastrow_orbital import GenericJastrowOrbitals
from ... import log


def set_jastrow_correlated(jastrow_type, nup, ndown, nmo, cuda, **kwargs):
    """Set the jastrow calculator

    Args:
        jastrow_type (str): name of the jastrow
        nup (int): number of up electrons
        ndown (int): number of down electrons
        cuda (bool): use cuda
    """

    valid_specialized_names = ['pade_jastrow', 'unity']
    #    'pade_jastrow_(n)',
    #    'scaled_pade_jastrow']

    # load specialized jastrows
    if isinstance(jastrow_type, str):
        if jastrow_type == 'pade_jastrow':
            return PadeJastrowOrbital(nup, ndown, nmo, w=1., cuda=cuda)

        elif jastrow_type == 'unity':
            return UnityJastrowOrbital(nup, ndown, nmo, cuda=cuda)

        else:
            log.info(
                '   Error : Specialized Jastrow form not recognized. Options are :')
            for n in valid_specialized_names:
                log.info('         : {0}', n)
            raise ValueError('Jastrow type not supported')

    # load generic jastrow without args
    elif issubclass(jastrow_type, torch.nn.Module):
        return GenericJastrowOrbitals(nup, ndown, nmo, jastrow_type, cuda, **kwargs)

    # default
    else:
        log.info(
            '   Error : Jastrow type not recognized. If string it should be either :')
        for n in valid_specialized_names:
            log.info('         : {0}', n)
        log.info(
            '         : if generic jastrow it should be subclass nn.Module')
        raise ValueError('Jastrow type not supported')
