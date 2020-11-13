import torch
from torch import nn

from ...utils import register_extra_attributes
from .two_body_jastrow_base import TwoBodyJastrowFactorBase


class PadeJastrowOrbital(TwoBodyJastrowFactorBase):

    def __init__(self, nup, ndown, nmo, w=1., cuda=False):
        r"""Computes Pade jastrow factor per MO

        .. math::
            J = \prod_{i<j} \exp(B_{ij}) \quad \quad \\text{with} \quad \quad
            B^k_{ij} = \\frac{w_0 r_{i,j}}{1 + w_k r_{i,j}}

        Args:
            nup ([type]): [description]
            down ([type]): [description]
            w ([type]): [description]
            cuda (bool, optional): [description]. Defaults to False.
        """

        super(PadeJastrowOrbital, self).__init__(nup, ndown, cuda)

        self.weight = nn.Parameter(
            w*torch.ones(nmo), requires_grad=True)
        register_extra_attributes(self, ['weight'])
        self.nmo = nmo
        self.static_weight = self.get_static_weight()

    def _get_jastrow_elements(self, r):
        r"""Get the elements of the jastrow matrix :
        .. math::
            out_{k,i,j} = \exp{ \frac{w r_{i,j}}{1+w_k r_{i,j}} }

        where k runs over the MO

        Args:
            r (torch.tensor): matrix of the e-e distances
                            Nbatch x Nelec_pair

        Returns:
            torch.tensor: matrix fof the jastrow elements
                        Nmo x Nbatch x Nelec_pair
        """
        return torch.exp(self._compute_kernel(r))

    def _compute_kernel(self, r):
        """ Get the jastrow kernel.
        .. math::
            B_{ij} = \frac{b r_{i,j}}{1+b'r_{i,j}}

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec_pair

        Returns:
            torch.tensor: matrix of the jastrow kernels
                          Nmo x Nbatch x Nelec_pair
        """

        denom = 1. / \
            (1. + self.weight.view(self.nmo, 1, 1) * r.unsqueeze(0))
        return self.static_weight * r * denom

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
                              Nbatch x Nelec_pair
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec_pair

        Returns:
            torch.tensor: matrix fof the derivative of the jastrow elements
                          Nmo x Nbatch x Ndim x  Nelec_pair
        """

        # convert all tensors to size compatible with output size
        # i.e. 4 dimensional tensors
        w = self.weight.view(self.nmo, 1, 1, 1)
        r_ = r.unsqueeze(0).unsqueeze(2)
        dr_ = dr[None, ...]

        denom = 1. / (1.0 + w * r_)
        a = self.static_weight * dr_ * denom
        b = - self.static_weight * w * r_ * dr_ * denom**2

        return (a + b)

    def _get_second_der_jastrow_elements(self, r, dr, d2r):
        """Get the elements of the pure 2nd derivative of the jastrow kernels
        wrt to the first electron

        .. math ::

            d^2 B_{ij} / d k_i^2 =  d^2 B_{ij} / d k_j^2 = d^2 B_{ji} / d k_i^2

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec_pair
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec_pair
            d2r (torch.tensor): matrix of the 2nd derivative of 
                                the e-e distances
                              Nbatch x Ndim x Nelec_pair

        Returns:
            torch.tensor: matrix fof the pure 2nd derivative of 
                          the jastrow elements
                          Nmo x Nbatch x Ndim x Nelec_pair
        """

        # convert all tensors to size compatible with output size
        # i.e. 4 dimensional tensors
        w = self.weight.view(self.nmo, 1, 1, 1)
        r_ = r.unsqueeze(0).unsqueeze(2)
        dr_ = dr[None, ...]
        d2r_ = d2r[None, ...]

        denom = 1. / (1.0 + w * r_)
        denom2 = denom**2
        dr_square = dr_*dr_

        a = self.static_weight * d2r_ * denom
        b = -2 * self.static_weight * w * dr_square * denom2
        c = - self.static_weight * w * r_ * d2r_ * denom2
        d = 2 * self.static_weight * w**2 * r_ * dr_square * denom**3

        e = self._get_der_jastrow_elements(r, dr)

        return a + b + c + d + e**2
