from .pade_jastrow import PadeJastrow
from .pade_jastrow_polynomial import PadeJastrowPolynomial
from .scaled_pade_jastrow import ScaledPadeJastrow

from .pade_jastrow_orbital import PadeJastrowOrbital

from ... import log


def set_jastrow(jastrow_type, nup, ndown, cuda):
    """Set the jastrow calculator

    Args:
        jastrow_type (str): name of the jastrow
        nup (int): number of up electrons
        ndown (int): number of down electrons
        cuda (bool): use cuda
    """

    if jastrow_type == 'pade_jastrow':
        return PadeJastrow(nup, ndown, w=1., cuda=cuda)

    elif jastrow_type.startswith('pade_jastrow('):
        order = int(jastrow_type.split('(')[1][0])
        return PadeJastrowPolynomial(nup, ndown, order, cuda=cuda)

    elif jastrow_type == 'scaled_pade_jastrow':
        return ScaledPadeJastrow(nup, ndown, w=1., kappa=0.6, cuda=cuda)

    else:
        valid_names = ['pade_jastrow',
                       'pade_jastrow_(n)',
                       'scaled_pade_jastrow']
        log.info(
            '   Error : Jastrow form not recognized. Options are :')
        for n in valid_names:
            log.info('         : {0}', n)
        raise ValueError('Jastrow type not supported')


def set_jastrow_correlated(jastrow_type, nup, ndown, nmo, cuda):
    """Set the jastrow calculator

    Args:
        jastrow_type (str): name of the jastrow
        nup (int): number of up electrons
        ndown (int): number of down electrons
        cuda (bool): use cuda
    """

    if jastrow_type == 'pade_jastrow':
        return PadeJastrowOrbital(nup, ndown, nmo, w=1., cuda=cuda)

    else:
        valid_names = ['pade_jastrow']
        #    'pade_jastrow_(n)',
        #    'scaled_pade_jastrow']
        log.info(
            '   Error : Jastrow form not recognized. Options are :')
        for n in valid_names:
            log.info('         : {0}', n)
        raise ValueError('Jastrow type not supported')
