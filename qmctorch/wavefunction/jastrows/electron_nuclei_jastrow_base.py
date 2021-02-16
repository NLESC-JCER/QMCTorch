import torch
from torch import nn
from .electron_nuclei_distance import ElectronNucleiDistance
import itertools


class ElectronNucleiJastrowFactorBase(nn.Module):

    def __init__(self, nup, ndown, atomic_pos, cuda=False):
        r"""Base class for two el-nuc jastrow of the form:

        .. math::
            J = \prod_{a,i} \exp(A(r_{ai}))

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            atomic_pos (tensor): positions of the atoms
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super(ElectronNucleiJastrowFactorBase, self).__init__()

        self.nup = nup
        self.ndown = ndown
        self.nelec = nup + ndown
        self.atoms = atomic_pos
        self.natoms = atomic_pos.shape[0]
        self.ndim = 3

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

        self.edist = ElectronNucleiDistance(
            self.nelec, self.atoms, self.ndim)

    def forward(self, pos, derivative=0, jacobian=True):
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
                          derivative = 0  (Nmo) x Nbatch x 1
                          derivative = 1  (Nmo) x Nbatch x Nelec (for jacobian = True)
                          derivative = 1  (Nmo) x Nbatch x Ndim x Nelec (for jacobian = False)
                          derivative = 2  (Nmo) x Nbatch x Nelec
        """

        size = pos.shape
        assert size[1] == self.nelec * self.ndim
        nbatch = size[0]

        r = self.edist(pos)
        jast = self._get_jastrow_elements(r)

        if derivative == 0:
            return jast.prod(-1).prod(-1).unsqueeze(-1)

        elif derivative == 1:
            dr = self.edist(pos, derivative=1)
            return self._jastrow_derivative(r, dr, jast, jacobian)

        elif derivative == 2:

            dr = self.edist(pos, derivative=1)
            d2r = self.edist(pos, derivative=2)

            return self._jastrow_second_derivative(r, dr, d2r, jast)

        elif derivative == [0, 1, 2]:

            dr = self.edist(pos, derivative=1)
            d2r = self.edist(pos, derivative=2)

            return(jast.prod(-1).prod(-1).unsqueeze(-1),
                   self._jastrow_derivative(r, dr, jast, jacobian),
                   self._jastrow_second_derivative(r, dr, d2r, jast))

    def _jastrow_derivative(self, r, dr, jast, jacobian):
        """Compute the value of the derivative of the Jastrow factor

        Args:
            r (torch.tensor): ee distance matrix Nbatch x Nelec x Nelec
            jast (torch.tensor): values of the jastrow elements
                                 Nbatch x Nelec x Natom

        Returns:
            torch.tensor: gradient of the jastrow factors
                          Nbatch x Ndim x Nelec
        """
        nbatch = r.shape[0]
        if jacobian:

            prod_val = jast.view(nbatch, -1).prod(-1, keepdim=True)
            djast = self._get_der_jastrow_elements(r, dr).sum((1, 3))

        else:
            prod_val = jast.view(
                nbatch, -1).prod(-1, keepdim=True).unsqueeze(-1)
            djast = self._get_der_jastrow_elements(r, dr).sum(3)

        return djast * prod_val

    def _jastrow_second_derivative(self, r, dr, d2r, jast):
        """Compute the value of the pure 2nd derivative of the Jastrow factor

        Args:
            r (torch.tensor): ee distance matrix Nbatch x Nelec x Nelec
            jast (torch.tensor): values of the ajstrow elements
                                 Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: diagonal hessian of the jastrow factors
                          Nbatch x Nelec x Ndim
        """
        nbatch = r.shape[0]

        # pure second derivative terms
        prod_val = jast.view(nbatch, -1).prod(-1, keepdim=True)

        d2jast = self._get_second_der_jastrow_elements(
            r, dr, d2r).sum((1, 3))

        # mixed terms
        djast = self._get_der_jastrow_elements(r, dr)

        # add partial derivative
        hess_jast = d2jast + 2 * \
            self.partial_derivative(djast)

        return hess_jast * prod_val

    def partial_derivative(self, djast):
        """Get the sum of the  product of the mixed
           second deriative terms

        .. math ::

            d B_{ij} / d x_i * d B_{kl} / d x_k

        Args:
            djast (torch.tensor): first derivative of the jastrow kernels

        Returns:
            torch.tensor:
        """

        # get the index of pairs
        idx = list(itertools.combinations(range(self.natoms), 2))

        # compute the sum of the product
        out = djast[..., idx].prod(-1).sum(-1)

        # return the sum over the dims
        return out.sum(1)

    def _get_jastrow_elements(self, r):
        r"""Get the elements of the jastrow matrix :
        .. math::
            out_{i,j} = \exp{ U(r_{ij}) }

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the jastrow elements
                          Nbatch x Nelec x Nelec
        """
        raise NotImplementedError('Jastrow element not implemented')

    def _get_der_jastrow_elements(self, r, dr):
        """Get the elements of the derivative of the jastrow kernels
        wrt to the first electrons

        .. math::

            d B_{ij} / d k_i =  d B_{ij} / d k_j  = - d B_{ji} / d k_i

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the derivative of the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """
        raise NotImplementedError(
            'Jastrow derivative not implemented')

    def _get_second_der_jastrow_elements(self, r, dr, d2r):
        """Get the elements of the pure 2nd derivative of the jastrow kernels
        wrt to the first electron

        .. math ::

            d^2 B_{ij} / d k_i^2 =  d^2 B_{ij} / d k_j^2 = d^2 B_{ji} / d k_i^2

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec
            d2r (torch.tensor): matrix of the 2nd derivative of
                                the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the pure 2nd derivative of
                          the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """
        raise NotImplementedError(
            'Jastrow second derivative not implemented')
