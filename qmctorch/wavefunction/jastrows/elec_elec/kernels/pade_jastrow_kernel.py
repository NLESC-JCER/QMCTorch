import torch
from torch import nn

from .....utils import register_extra_attributes
from .jastrow_kernel_electron_electron_base import JastrowKernelElectronElectronBase


class PadeJastrowKernel(JastrowKernelElectronElectronBase):

    def __init__(self, nup, ndown, cuda, w=1.):
        """Computes the Simple Pade-Jastrow factor

        .. math::
            B_{ij} = \\frac{w_0 r_{ij}}{1 + w r_{ij}}

        where :math:`w_0` equals 0.5 for parallel spin and 0.25 for antiparallel spin

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            cuda (bool): Turns GPU ON/OFF.
            w (float, optional): Value of the variational parameter.
                                 Defaults to 1.

        """

        super().__init__(nup, ndown, cuda)

        self.weight = nn.Parameter(torch.as_tensor([w]),
                                   requires_grad=True)
        register_extra_attributes(self, ['weight'])

        self.static_weight = self.get_static_weight()
        self.requires_autograd = False

    def get_static_weight(self):
        """Get the matrix of static weights

        Returns:
            torch.tensor: matrix of the static weights
        """

        bup = torch.cat((0.25 * torch.ones(self.nup, self.nup), 0.5 *
                         torch.ones(self.nup, self.ndown)), dim=1)

        bdown = torch.cat((0.5 * torch.ones(self.ndown, self.nup), 0.25 *
                           torch.ones(self.ndown, self.ndown)), dim=1)

        static_weight = torch.cat((bup, bdown), dim=0).to(self.device)

        mask_tri_up = torch.triu(torch.ones_like(
            static_weight), diagonal=1).type(torch.BoolTensor).to(self.device)
        static_weight = static_weight.masked_select(mask_tri_up)

        return static_weight

    def forward(self, r):
        """ Get the jastrow kernel.

        .. math::
            B_{ij} = \\frac{w_0 r_{i,j}}{1+w r_{i,j}}

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

            \\frac{d B_{ij}}{d k_i} =  \\frac{d B_{ij}}{ d k_j }  = - \\frac{d B_{ji}}{d k_i}

        .. math::
            \\text{out}_{k,i,j} = A1 + A2

        .. math::
            A1_{kij} = w0 \\frac{dr_{ij}}{dk_i}  \\frac{1}{1 + w r_{ij}}

        .. math::
            A2_{kij} = - w0 w' r_{ij} \\frac{dr_{ij}}{dk_i} \\frac{1}{1 + w r_{ij}}^2

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
        b = -self.static_weight * self.weight * r_ * dr * denom**2

        return (a + b)

    def compute_second_derivative(self, r, dr, d2r):
        """Get the elements of the pure 2nd derivative of the jastrow kernels
        wrt to the first electron

        .. math ::

            \\frac{d^2 B_{ij}}{d k_i^2} =  \\frac{d^2 B_{ij}}{d k_j^2} = \\frac{d^2 B_{ji}}{ d k_i^2}

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec
            d2r (torch.tensor): matrix of the 2nd derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the pure 2nd derivative of
                          the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """

        r_ = r.unsqueeze(1)
        denom = 1. / (1.0 + self.weight * r_)
        denom2 = denom**2
        dr_square = dr * dr

        a = self.static_weight * d2r * denom
        b = -2 * self.static_weight * self.weight * dr_square * denom2
        c = -self.static_weight * self.weight * r_ * d2r * denom2
        d = 2 * self.static_weight * self.weight**2 * r_ * dr_square * denom**3

        return a + b + c + d
