import torch
from torch import nn
from .backflow_kernel_base import BackFlowKernelnBase


class BackFlowKernelnInverse(BackFlowKernelnBase):

    def __init__(self, mol, cuda):
        """Compute the back flow kernel, i.e. the function
        f(rij) where rij is the distance between electron i and j
        This kernel is used in the backflow transformation
        .. math:
            q_i = r_i + \\sum_{j\\neq i} f(r_{ij}) (r_i-r_j)
        """
        super().__init__(mol, cuda)
        self.weight = nn.Parameter(
            torch.as_tensor([1E-4])).to(self.device)

    def forward(self, ree, derivative=0):
        """Computes the desired values of the kernel
         Args:
            ree (torch.tensor): e-e distance Nbatch x Nelec x Nelec
            derivative (int): derivative requried 0, 1, 2

        Returns:
            torch.tensor : f(r) Nbatch x Nelec x Nelec
        """

        if derivative == 0:
            return self._backflow_kernel(ree)

        elif derivative == 1:
            return self._backflow_kernel_derivative(ree)

        elif derivative == 2:
            return self._backflow_kernel_second_derivative(ree)

        else:
            raise ValueError(
                'derivative of the kernel must be 0, 1 or 2')

    def _backflow_kernel(self, ree):
        """Computes the backflow kernel:

        .. math:
            \\eta(r_{ij}) = \\frac{u}{r_{ij}}

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
            \\frac{d}{dr_{ij} \\eta(r_{ij}) = -u r_{ij}^{-2}

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
            \\frac{d}{dr_{ij} \\eta(r_{ij}) = -u r_{ij}^{-2}

        Args:
            ree (torch.tensor): e-e distance Nbatch x Nelec x Nelec

        Returns:
            torch.tensor : f''(r) Nbatch x Nelec x Nelec
        """

        eye = torch.eye(self.nelec, self.nelec).to(self.device)
        invree = (1./(ree+eye) - eye)
        return 2 * self.weight * invree * invree * invree
