import torch
from torch import nn
from torch.autograd import grad
import numpy as np
from ...utils import register_extra_attributes, diagonal_hessian
from .two_body_jastrow_base import TwoBodyJastrowFactorBase


class FullyConnectedJastrow(torch.nn.Module):

    def __init__(self):
        """Defines a fully connected jastrow factors."""

        super(FullyConnectedJastrow, self).__init__()

        self.cusp_weights = None

        self.fc1 = nn.Linear(1, 128, bias=False)
        self.fc2 = nn.Linear(128, 256, bias=False)
        self.fc3 = nn.Linear(256, 1, bias=False)

    @staticmethod
    def get_cusp_weights(npairs):
        """Computes the cusp bias

        Args:
            npairs (int): number of elec pairs
        """
        nelec = int(0.5 * (-1 + np.sqrt(1+8*npairs)))
        weights = torch.zeros(npairs)

        spin = torch.ones(nelec)
        spin[:int(nelec/2)] = -1
        ip = 0
        for i1 in range(nelec):
            for i2 in range(i1, nelec):
                if spin[i1] == spin[i2]:
                    weights[ip] = 0.25
                else:
                    weights[ip] = 0.5
                ip += 1
        return weights

    def forward(self, x):
        """Compute the values of the individual f_ij=f(r_ij)

        Args:
            x (torch.tensor): e-e distance Nbatch, Nele_pairs

        Returns:
            torch.tensor: values of the f_ij
        """
        nbatch, npairs = x.shape

        if self.cusp_weights is None:
            self.cusp_weights = self.get_cusp_weights(npairs)

        # reshape the input so that all elements are considered
        # independently of each other
        x = x.reshape(-1, 1)

        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = 2*x
        # reshape to the original shape
        x = x.reshape(nbatch, npairs)

        # add the cusp weight
        # x = x + self.cusp_weights

        return 2*x


class GenericJastrowOrbitals(TwoBodyJastrowFactorBase):

    def __init__(self, nup, ndown, nmo, JastrowFunctions, cuda=False, **kwargs):
        r"""Computes Pade jastrow factor per MO

        .. math::
            J = \prod_{i<j} \exp(B_{ij}) \quad \quad \\text{with} \quad \quad
            B^k_{ij} = \\frac{w_0 r_{i,j}}{1 + w_k r_{i,j}}

        Args:
            nup ([type]): [description]
            down ([type]): [description]
            cuda (bool, optional): [description]. Defaults to False.
        """

        super(GenericJastrowOrbitals, self).__init__(
            nup, ndown, cuda)
        self.nmo = nmo
        self.jastrow_functions = nn.ModuleList(
            [JastrowFunctions(**kwargs) for imo in range(self.nmo)])

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

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec_pair

        Returns:
            torch.tensor: matrix of the jastrow kernels
                          Nmo x Nbatch x Nelec_pair
        """
        out = None
        for jast in self.jastrow_functions:
            jvals = jast(r).unsqueeze(0)
            if out is None:
                out = jvals
            else:
                out = torch.cat((out, jvals), axis=0)
        return out

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

        out = None
        for jast in self.jastrow_functions:

            kernel = jast(r)
            ker_grad = self._grads(kernel, r)
            ker_grad = ker_grad.unsqueeze(1) * dr
            ker_grad = ker_grad.unsqueeze(0).detach().clone()

            if out is None:
                out = ker_grad
            else:
                out = torch.cat((out, ker_grad), axis=0)

        return out

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
        dr2 = dr * dr

        out = None
        for jast in self.jastrow_functions:

            jvals = jast(r)
            jhess, jgrads = self._hess(jvals, r)
            jhess = jhess.unsqueeze(
                1) * dr2 + jgrads*unsqueeze(1) * d2r
            jhess = jhess.unsqueeze(0)

            if out is None:
                out = jhess
            else:
                out = torch.cat((out, jhess))

        return out

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

        gval = grad(val, pos,
                    grad_outputs=torch.ones_like(val),
                    create_graph=True)[0]

        hval = grad(gval, pos,
                    grad_outputs=torch.ones_like(gval))[0]

        return hval, gval
