
import torch
from torch import nn
from torch.autograd import grad
from .kernels.jastrow_kernel_electron_electron_base import JastrowKernelElectronElectronBase


class OrbitalDependentJastrowKernel(JastrowKernelElectronElectronBase):

    def __init__(self, nup, ndown, nmo, cuda,
                 jastrow_kernel, kernel_kwargs={}):
        """Transform a kernel into a orbital dependent kernel

        Args:
            nup (int): number of spin up electrons
            ndown (int): number of spin down electron
            nmo (int): number of orbital
            cuda (bool): use GPUs
            jastrow_kernel (kernel function): kernel to be used
            kernel_kwargs (dict): keyword arguments of the kernel
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

        if r.requires_grad == False:
            r.requires_grad = True

        with torch.enable_grad():

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

        if r.requires_grad == False:
            r.requires_grad = True

        with torch.enable_grad():

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
    def _grads(val, r):
        """Get the gradients of the jastrow values
        of a given orbital terms

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec_pair

        Returns:
            torch.tensor: gradients of the values wrt to ee distance
        """
        return grad(val, r, grad_outputs=torch.ones_like(val))[0]

    @staticmethod
    def _hess(val, r):
        """get the hessian of the jastrow values.
        of a given orbital terms
        Warning thos work only because the orbital term are dependent
        of a single rij term, i.e. fij = f(rij)


        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec_pair

        Returns:
            torch.tensor: second derivative of the values wrt to ee distance
        """

        gval = grad(val, r,
                    grad_outputs=torch.ones_like(val),
                    create_graph=True)[0]

        hval = grad(gval, r,
                    grad_outputs=torch.ones_like(gval))[0]

        return hval, gval
