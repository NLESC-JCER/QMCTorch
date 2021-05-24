
import torch
from torch import nn
from torch.autograd import grad
from .kernels.jastrow_kernel_electron_electron_base import JastrowKernelElectronElectronBase


class OrbitalDependentJastrowKernel(JastrowKernelElectronElectronBase):

    def __init__(self, nup, ndown, nmo, cuda,
                 jastrow_kernel, kernel_kwargs={}):
        r"""Computes generic jastrow factor per MO

        .. math::
            J = \prod_{i<j} \exp(B_{ij}) \quad \quad \\text{with} \quad \quad
            B^k_{ij} = \\frac{w_0 r_{i,j}}{1 + w_k r_{i,j}}

        Args:
            nup ([type]): [description]
            down ([type]): [description]
            cuda (bool, optional): [description]. Defaults to False.
        """

        super().__init__(nup, ndown, cuda)
        self.nmo = nmo
        self.jastrow_functions = nn.ModuleList(
            [jastrow_kernel(nup, ndown, cuda, **kernel_kwargs) for _ in range(self.nmo)])

    def forward(self, r):
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

    def compute_second_derivative(self, r, dr, d2r):
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

            kernel = jast(r)
            ker_hess, ker_grad = self._hess(kernel, r)

            jhess = (ker_hess).unsqueeze(1) * \
                dr2 + ker_grad.unsqueeze(1) * d2r

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
