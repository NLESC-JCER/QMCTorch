import torch
from torch import nn
from .electron_distance import ElectronDistance
from .pade_jastrow import PadeJastrow
from ..utils import register_extra_attributes
import itertools
from time import time


class ScaledPadeJastrow(PadeJastrow):

    def __init__(self, nup, ndown, w=1., kappa=0.6, cuda=False):
        r"""Computes the Simple Pade-Jastrow factor

        .. math::
            J = \prod_{i<j} \exp(B_{ij}) \quad \quad \\text{with} \quad \quad
            B_{ij} = \\frac{w_0 r_{i,j}}{1 + w r_{i,j}}

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            w (float, optional): Value of the variational parameter. Defaults to 1.
            kappa (float, optional): value of the scale parameter. Defaults to 0.6.
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super(ScaledPadeJastrow, self).__init__(
            nup, ndown, w, cuda)

        self.weight = nn.Parameter(
            torch.tensor([w]), requires_grad=True)
        self.edist.kappa = kappa
        self.static_weight = self.get_static_weight()
        register_extra_attributes(self, ['weight'])

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
        ur = self.edist.get_scaled_distance(r)
        return self.static_weight * ur / (1.0 + self.weight * ur)

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

        u = self.edist.get_scaled_distance(r).unsqueeze(1)
        du = self.edist.get_der_scaled_distance(r, dr)

        denom = 1. / (1.0 + self.weight * u)

        a = self.static_weight * du * denom
        b = - self.static_weight * self.weight * u * du * denom**2

        return (a + b)

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

        u = self.edist.get_scaled_distance(r).unsqueeze(1)
        du = self.edist.get_der_scaled_distance(r, dr)
        d2u = self.edist.get_second_der_scaled_distance(r, dr, d2r)

        denom = 1. / (1.0 + self.weight * u)
        denom2 = denom**2
        du_square = du*du

        a = self.static_weight * d2u * denom
        b = -2 * self.static_weight * self.weight * du_square * denom2
        c = - self.static_weight * self.weight * u * d2u * denom2
        d = 2 * self.static_weight * self.weight**2 * u * du_square * denom**3

        e = self._get_der_jastrow_elements(r, dr)

        return a + b + c + d + e**2
