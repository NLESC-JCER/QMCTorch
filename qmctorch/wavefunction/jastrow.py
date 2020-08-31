from .jastrow_factors import PadeJastrow
from .jastrow_factors import ScaledPadeJastrow
from .jastrow_factors import PadeJastrowPolynomial
from .jastrow_factors import ScaledPadeJastrowPolynomial
from .. import log


def Jastrow(mol, jastrow_type, cuda=False):
    """Main routine to define the  jastrow factor

    Args:
        mol (Molecule): instance of a Molecule object
        jastrow_type (str): type of jastrow factor required
        cuda (bool, optional): Use CUDA or not. Defaults to False.

    Raises:
        ValueError: if jastrow type not recognized
    """

    valid_names = ['pade_jastrow',
                   'pade_jastrow(n)',
                   'scaled_pade_jastrow',
                   'scaled_pade_jastrow(n)']

    if jastrow_type == 'pade_jastrow':
        return PadeJastrow(mol.nup, mol.ndown,
                           w=1., cuda=cuda)

    elif jastrow_type.startswith('pade_jastrow('):
        order = int(jastrow_type.split('(')[1][0])
        return PadeJastrowPolynomial(
            mol.nup, mol.ndown, order, cuda=cuda)

    elif jastrow_type == 'scaled_pade_jastrow':
        return ScaledPadeJastrow(mol.nup, mol.ndown,
                                 w=1., kappa=0.6, cuda=cuda)

    elif jastrow_type.startswith('scaled_pade_jastrow('):
        order = int(jastrow_type.split('(')[1][0])
        return ScaledPadeJastrowPolynomial(
            mol.nup, mol.ndown, order, kappa=0.6, cuda=cuda)

    else:
        log.info(
            '   Error : Jastrow form not recognized. Options are :')
        for n in valid_names:
            log.info('         : {0}', n)
        raise ValueError('Jastrow type not supported')
