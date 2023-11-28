import torch
from torch import nn
from .backflow_kernel_base import BackFlowKernelBase


class BackFlowKernelInverse(BackFlowKernelBase):

    def __init__(self, mol, cuda=False):
        """Compute the back flow kernel, i.e. the function
        f(rij) where rij is the distance between electron i and j
        This kernel is used in the backflow transformation
        .. math:
            q_i = r_i + \\sum_{j\\neq i} f(r_{ij}) (r_i-r_j)

        with here :

        .. math:
            f(r_{ij) = \\frac{w}{r_{ij}
        """
        super().__init__(mol, cuda)
        self.weight = nn.Parameter(
            torch.as_tensor([1E-3]))  # .to(self.device)

    def _backflow_kernel(self, ree):
        """Computes the backflow kernel:

        .. math:
            \\eta(r_{ij}) = \\frac{w}{r_{ij}}

        Args:
            r (torch.tensor): e-e distance Nbatch x Nelec x Nelec

        Returns:
            torch.tensor : f(r) Nbatch x Nelec x Nelec
        """

        eye = torch.eye(self.nelec, self.nelec).to(self.device)
        mask = torch.ones_like(ree) - eye
        return self.weight * mask * (1./(ree+eye) - eye)

    def _backflow_kernel_derivative(self, ree):
        """Computes the derivative of the kernel function
            w.r.t r_{ij}
        .. math::
            \\frac{d}{dr_{ij} \\eta(r_{ij}) = -w r_{ij}^{-2}

        Args:
            ree (torch.tensor): e-e distance Nbatch x Nelec x Nelec

        Returns:
            torch.tensor : f'(r) Nbatch x Nelec x Nelec
        """

        eye = torch.eye(self.nelec, self.nelec).to(self.device)
        invree = (1./(ree+eye) - eye)
        return - self.weight * invree * invree

    def _backflow_kernel_second_derivative(self, ree):
        """Computes the derivative of the kernel function
            w.r.t r_{ij}
        .. math::
            \\frac{d^2}{dr_{ij}^2} \\eta(r_{ij}) = 2 w r_{ij}^{-3}

        Args:
            ree (torch.tensor): e-e distance Nbatch x Nelec x Nelec

        Returns:
            torch.tensor : f''(r) Nbatch x Nelec x Nelec
        """

        eye = torch.eye(self.nelec, self.nelec).to(self.device)
        invree = (1./(ree+eye) - eye)
        return 2 * self.weight * invree * invree * invree
