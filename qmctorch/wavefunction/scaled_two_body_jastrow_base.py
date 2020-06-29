import torch
from torch import nn
from .electron_distance import ElectronDistance
from . two_body_jastrow_base import TwoBodyJastrowFactorBase
from ..utils import register_extra_attributes
import itertools
from time import time


class ScaledTwoBodyJastrowFactorBase(TwoBodyJastrowFactorBase):

    def __init__(self, nup, ndown, kappa=0.6, cuda=False):
        r"""Base class for scaled two body jastrow of the form:

        .. math::
            J = \prod_{i<j} \exp(B(U_{ij}))
            U = \frac{1+e^{-kr}}{k}      

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super(ScaledPadeJastrow, self).__init__(nup, ndown, cuda)
        self.kappa = kappa
        register_extra_attributes(self, ['kappa'])

    def _get_scaled_distance(self, r):
        """compute the scaled distance 
        .. math::
            u = \frac{1+e^{-kr}}{k}      

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: values of the scaled distance
                          Nbatch, Nelec, Nelec
        """
        return (1. - torch.exp(-self.kappa * r))/self.kappa

    def _get_der_scaled_distance(self, r, dr):
        """Returns the derivative of the scaled distances

        Args:
            r (torch.tensor): unsqueezed matrix of the e-e distances
                              Nbatch x Nelec x Nelec

            dr (torch.tensor): matrix of the derivative of the e-e distances
                               Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor : deriative of the scaled distance
                          Nbatch x Ndim x Nelec x Nelec
        """
        return dr * torch.exp(-self.kappa * r.unsqueeze(1))

    def _get_second_der_scaled_distance(self, r, dr, d2r):
        """computes the second derivative of the scaled distances

        Args:
            r (torch.tensor): unsqueezed matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec
            d2r (torch.tensor): matrix of the 2nd derivative of
                                the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor : second deriative of the scaled distance
                          Nbatch x Ndim x Nelec x Nelec
        """
        return (d2r - self.kappa * dr * dr) * torch.exp(-self.kappa*r.unsqueeze(1))
