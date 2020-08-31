from .jastrow.pade_jastrow import PadeJastrow
from .jastrow.scaled_pade_jastrow import ScaledPadeJastrow
from .jastrow.pade_jastrow_polynomial import PadeJastrowPolynomial
from .jastrow.scaled_pade_jastrow_polynomial import ScaledPadeJastrowPolynomial
from .. import log


class Jastrow():

    def __init__(self, mol, jastrow_type, cuda=False):
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
            self._jastrow_calc = PadeJastrow(mol.nup, mol.ndown,
                                             w=1., cuda=cuda)

        elif jastrow_type.startswith('pade_jastrow('):
            order = int(jastrow_type.split('(')[1][0])
            self._jastrow_calc = PadeJastrowPolynomial(
                mol.nup, mol.ndown, order, cuda=cuda)

        elif jastrow_type == 'scaled_pade_jastrow':
            self._jastrow_calc = ScaledPadeJastrow(mol.nup, mol.ndown,
                                                   w=1., kappa=0.6, cuda=cuda)

        elif jastrow_type.startswith('scaled_pade_jastrow('):
            order = int(jastrow_type.split('(')[1][0])
            self._jastrow_calc = ScaledPadeJastrowPolynomial(
                mol.nup, mol.ndown, order, kappa=0.6, cuda=cuda)

        else:
            log.info(
                '   Error : Jastrow form not recognized. Options are :')
            for n in valid_names:
                log.info('         : {0}', n)
            raise ValueError('Jastrow type not supported')

    def __call__(self, pos, derivative=0, jacobian=True):
        """Compute the Jastrow factors.

        Args:
            pos (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0,1,2,).
                            Defaults to 0.
            jacobian (bool, optional): Return the jacobian (i.e. the sum of
                                       the derivatives) or the individual
                                       terms. Defaults to True.
                                       False only for derivative=1

        Returns:
            torch.tensor: value of the jastrow parameter for all confs
        """
        return self._jastrow_calc(pos, derivative, jacobian)
