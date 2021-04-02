import torch
from torch import nn
from torch.autograd import grad
from ....utils import register_extra_attributes
from .electron_nuclei_base import ElectronNucleiBase


class ElectronNucleiGeneric(ElectronNucleiBase):
    def __init__(self, nup, ndown, atoms, JastrowFunction, cuda, **kwargs):
        r"""Computes the Simple Pade-Jastrow factor

        .. math::
            J = \prod_{i<j} \exp(B_{ij}) \quad \quad \\text{with} \quad \quad
            B_{ij} = \\frac{w_0 r_{i,j}}{1 + w r_{i,j}}

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            atoms (torch.tensor): atomic positions of the atoms
            w (float, optional): Value of the variational parameter. Defaults to 1..
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super(ElectronNucleiGeneric, self).__init__(
            nup, ndown, atoms, cuda)

        self.jastrow_function = JastrowFunction(**kwargs)

    def _get_jastrow_elements(self, r):
        r"""Get the elements of the jastrow matrix :
        .. math::
            out_{i,j} = \exp{ \frac{b r_{i,j}}{1+b'r_{i,j}} }

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the jastrow elements
                          Nbatch x Nelec x Nelec
        """
        return torch.exp(self._compute_kernel(r))

    def _compute_kernel(self, r):
        """ Get the jastrow kernel.
        .. math::
            B_{ij} = \frac{b r_{i,j}}{1+b'r_{i,j}}

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: matrix of the jastrow kernels
                          Nbatch x Nelec x Nelec
        """
        return self.jastrow_function(r)

    def _get_der_jastrow_elements(self, r, dr):
        """Get the elements of the derivative of the jastrow kernels
        wrt to the first electrons

        .. math::

            d B_{ij} / d k_i =  d B_{ij} / d k_j  = - d B_{ji} / d k_i

            out_{k,i,j} = A1 + A2
            A1_{kij} = w0 \frac{dr_{ij}}{dk_i} / (1 + w r_{ij})
            A2_{kij} = - w0 w' r_{ij} \frac{dr_{ij}}{dk_i} / (1 + w r_{ij})^2

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the derivative of the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """

        kernel = self.jastrow_function(r)
        ker_grad = self._grads(kernel, r)

        return ker_grad.unsqueeze(1) * dr

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

        dr2 = dr * dr

        kernel = self.jastrow_function(r)

        ker_hess, ker_grad = self._hess(kernel, r)

        jhess = (ker_hess).unsqueeze(1) * \
            dr2 + ker_grad.unsqueeze(1) * d2r

        return jhess

    @staticmethod
    def _grads(val, pos):
        """Get the gradients of the jastrow values
        of a given orbital terms

        Args:
            pos ([type]): [description]

        Returns:
            [type]: [description]
        """
        return grad(val, pos, grad_outputs=torch.ones_like(val))[0]

    @staticmethod
    def _hess(val, pos):
        """get the hessian of the jastrow values.
        of a given orbital terms
        Warning thos work only because the orbital term are dependent
        of a single rij term, i.e. fij = f(rij)

        Args:
            pos ([type]): [description]
        """

        gval = grad(val,
                    pos,
                    grad_outputs=torch.ones_like(val),
                    create_graph=True)[0]

        hval = grad(gval, pos, grad_outputs=torch.ones_like(gval))[0]

        return hval, gval
