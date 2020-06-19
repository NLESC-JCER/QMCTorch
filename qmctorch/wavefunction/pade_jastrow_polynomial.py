import torch
from torch import nn
from .electron_distance import ElectronDistance
from . two_body_jastrow_base import TwoBodyJastrowFactorBase
from ..utils import register_extra_attributes
import itertools
from time import time


class PadeJastrowPolynomial(TwoBodyJastrowFactorBase):

    def __init__(self, nup, ndown, order, cuda=False):
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
            w (float, optional): Value of the variational parameter. Defaults to 1..
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super(PadeJastrowPolynomial, self).__init__(nup, ndown, cuda)
        self.porder = order

        self.weight_a = nn.Parameter(
            0.1*torch.zeros(self.porder)).to(self.device)
        self.weight_a.requires_grad = True
        register_extra_attributes(self, ['weight_a'])

        self.weight_b = nn.Parameter(
            0.1*torch.zeros(self.porder)).to(self.device)
        self.weight_b.requires_grad = True
        register_extra_attributes(self, ['weight_b'])

        self.static_weight = self.get_static_weight()

    def get_static_weight(self):
        """Get the matrix of static weights

        Returns:
            torch.tensor: static weight (0.5 (0.25)for parallel(anti) spins
        """

        bup = torch.cat((0.25 * torch.ones(self.nup, self.nup), 0.5 *
                         torch.ones(self.nup, self.ndown)), dim=1)

        bdown = torch.cat((0.5 * torch.ones(self.ndown, self.nup), 0.25 *
                           torch.ones(self.ndown, self.ndown)), dim=1)

        static_weight = torch.cat((bup, bdown), dim=0).to(self.device)
        static_weight = static_weight.masked_select(self.mask_tri_up)

        return static_weight

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

        num, denom = self._compute_polynoms(r)
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

        num, denom = self._compute_polynoms(r)
        num = num.unsqueeze(1)
        denom = denom.unsqueeze(1)

        der_num, der_denom = self._compute_polynom_derivatives(r, dr)

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

        num, denom = self._compute_polynoms(r)
        num = num.unsqueeze(1)
        denom = denom.unsqueeze(1)
        print(denom)
        der_num, der_denom = self._compute_polynom_derivatives(r, dr)

        d2_num, d2_denom = self._compute_polynom_second_derivative(
            r, dr, d2r)

        return d2_num/denom - (2*der_num*der_denom + num*d2_denom)/(denom*denom) \
            + 2 * num*der_denom*der_denom/(denom*denom*denom)

    def _compute_polynoms(self, r):
        """Compute the num and denom polynomials.

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec

        Returns:
            torch.tensor, torch.tensor : p and q polynoms
                                         size Nbatch x Nelec x Nelec
        """

        num = self.static_weight * r
        denom = (1.0 + self.weight_b[0] * r)
        riord = r.clone()

        for iord in range(1, self.porder):
            riord = riord * r
            num += self.weight_a[iord] * riord
            denom += self.weight_b[iord] * riord

        return num, denom

    def _compute_polynom_derivatives(self, r, dr):
        """Computes the derivatives of the polynomials.

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor, torch.tensor : p and q polynoms derivatives 
                                         size Nbatch x Ndim x Nelec x Nelec

        """

        der_num = self.static_weight * dr
        der_denom = self.weight_b[0] * dr

        r_ = r.unsqueeze(1)
        riord = r.unsqueeze(1)

        for iord in range(1, self.porder):

            fact = (iord+1) * dr * riord
            der_num += self.weight_a[iord] * fact
            der_denom += self.weight_b[iord] * fact
            riord = riord * r_

        return der_num, der_denom

    def _compute_polynom_second_derivative(self, r, dr, d2r):
        """Computes the second derivative of the polynoms.

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec
            d2r (torch.tensor): matrix of the 2nd derivative of
                                the e-e distances
                              Nbatch x Ndim x Nelec x Nelec
        Returns:
            torch.tensor, torch.tensor : p and q polynoms derivatives 
                                         size Nbatch x Ndim x Nelec x Nelec

        """

        der_num = self.static_weight * d2r
        der_denom = self.weight_b[0] * d2r
        dr2 = dr*dr

        r_ = r.unsqueeze(1)
        rnm1 = r.unsqueeze(1)
        rnm2 = 1.

        for iord in range(1, self.porder):

            n = iord+1
            fact = n * (d2r * rnm1 + iord * dr2*rnm2)
            der_num += self.weight_a[iord] * fact
            der_denom += self.weight_b[iord] * fact

            rnm2 = rnm1
            rnm1 = rnm1 * r_

        return der_num, der_denom
