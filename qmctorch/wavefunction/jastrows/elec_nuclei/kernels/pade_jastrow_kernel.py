import torch
from torch import nn

from .....utils import register_extra_attributes
from .jastrow_kernel_electron_nuclei_base import JastrowKernelElectronNucleiBase


class PadeJastrowKernel(JastrowKernelElectronNucleiBase):

    def __init__(self, nup, ndown, atomic_pos, cuda, w=1.):
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

        super().__init__(nup, ndown, atomic_pos, cuda)

        self.weight = nn.Parameter(
            torch.as_tensor([w]), requires_grad=True).to(self.device)
        register_extra_attributes(self, ['weight'])

        self.static_weight = torch.as_tensor([1.]).to(self.device)
        self.requires_autograd = True

    def forward(self, r):
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
        return self.static_weight * r / (1.0 + self.weight * r)

    def compute_derivative(self, r, dr):
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

        r_ = r.unsqueeze(1)
        denom = 1. / (1.0 + self.weight * r_)
        a = self.static_weight * dr * denom
        b = - self.static_weight * self.weight * r_ * dr * denom**2

        return (a + b)

    def compute_second_derivative(self, r, dr, d2r):
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

        r_ = r.unsqueeze(1)
        denom = 1. / (1.0 + self.weight * r_)
        denom2 = denom**2
        dr_square = dr*dr

        a = self.static_weight * d2r * denom
        b = -2 * self.static_weight * self.weight * dr_square * denom2
        c = - self.static_weight * self.weight * r_ * d2r * denom2
        d = 2 * self.static_weight * self.weight**2 * r_ * dr_square * denom**3

        return a + b + c + d
