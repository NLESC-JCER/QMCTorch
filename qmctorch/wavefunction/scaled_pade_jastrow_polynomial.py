import torch
from torch import nn
from .electron_distance import ElectronDistance
from .pade_jastrow_polynomial import PadeJastrowPolynomial
from ..utils import register_extra_attributes
import itertools
from time import time


class ScaledPadeJastrowPolynomial(PadeJastrowPolynomial):

    def __init__(self, nup, ndown, order, kappa=0.6, weight_a=None, weight_b=None, cuda=False):
        r"""Computes the Simple Pade-Jastrow factor

        .. math::
            J = \prod_{i<j} \exp(B_{ij}) \quad \quad \\text{with} \quad \quad
            B_{ij} =  \\frac{P_{ij}}{Q_{ij}}

            P_{ij} = a_1 r_{i,j} + a_2 r_{ij}^2 + ....
            Q_{ij} = 1 + b_1 r_{i,j} + b_2 r_{ij}^2 + ...

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            order (int): degree of the polynomial
            kappa (float, optional): value of the scale parameter. Defaults to 0.6.
            weight_a (torch.tensor, optional): Value of the weight on the numerator
            weight_b (torch.tensor, optional): Value of the weight on the numerator
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super(ScaledPadeJastrowPolynomial,
              self).__init__(nup, ndown, order, weight_a, weight_b, cuda)
        self.porder = order
        self.edist.kappa = kappa
        self.set_variational_weights(weight_a, weight_b)

        self.static_weight = self.get_static_weight()

    def _compute_kernel(self, r):
        """ Get the jastrow kernel.
        .. math::
            B_{ij} = \\frac{P_{ij}}{Q_{ij}}

            P_{ij} = a_1 r_{i,j} + a_2 r_{ij}^2 + ....
            Q_{ij} = 1 + b_1 r_{i,j} + b_2 r_{ij}^2 + ...

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: matrix of the jastrow kernels
                          Nbatch x Nelec x Nelec
        """

        u = self.edist.get_scaled_distance(r)
        num, denom = self._compute_polynoms(u)
        return num / denom

    def _get_jastrow_elements(self, r):
        r"""Get the elements of the jastrow matrix :
        .. math::
            out_{i,j} = \exp{ U(r_{ij}) }

            U(r_{ij}) = \\frac{P_{ij}}{Q_{ij}}

            P_{ij} = a_1 r_{i,j} + a_2 r_{ij}^2 + ....
            Q_{ij} = 1 + b_1 r_{i,j} + b_2 r_{ij}^2 + ...

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the jastrow elements
                          Nbatch x Nelec x Nelec
        """
        return torch.exp(self._compute_kernel(r))

    def _get_der_jastrow_elements(self, r, dr):
        """Get the elements of the derivative of the jastrow kernels
        wrt to the first electrons

        .. math::

            d B_{ij} / d k_i =  d B_{ij} / d k_j  = - d B_{ji} / d k_i

            out_{k,i,j} = \\frac{P'Q - PQ'}{Q^2}

            P_{ij} = a_1 r_{i,j} + a_2 r_{ij}^2 + ....
            Q_{ij} = 1 + b_1 r_{i,j} + b_2 r_{ij}^2 +

            P'_{ij} = a_1 dr + a_2 2 r dr + a_r 3 dr r^2 + ....
            Q'_{ij} = b_1 dr + b_2 2 r dr + b_r 3 dr r^2 + ....

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the derivative of the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """

        u = self.edist.get_scaled_distance(r)
        num, denom = self._compute_polynoms(u)

        num = num.unsqueeze(1)
        denom = denom.unsqueeze(1)

        du = self.edist.get_der_scaled_distance(r, dr)

        der_num, der_denom = self._compute_polynom_derivatives(u, du)

        return (der_num * denom - num * der_denom)/(denom*denom)

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

        u = self.edist.get_scaled_distance(r)
        du = self.edist.get_der_scaled_distance(r, dr)
        d2u = self.edist.get_second_der_scaled_distance(r, dr, d2r)

        num, denom = self._compute_polynoms(u)
        num = num.unsqueeze(1)
        denom = denom.unsqueeze(1)

        der_num, der_denom = self._compute_polynom_derivatives(u, du)

        d2_num, d2_denom = self._compute_polynom_second_derivative(
            u, du, d2u)

        out = d2_num/denom - (2*der_num*der_denom + num*d2_denom)/(
            denom*denom) + 2 * num*der_denom*der_denom/(denom*denom*denom)

        return out + self._get_der_jastrow_elements(r, dr)**2
