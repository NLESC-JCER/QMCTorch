import torch
from torch import nn
from .backflow_kernel_base import BackFlowKernelBase
from .....scf import Molecule


class BackFlowKernelSquare(BackFlowKernelBase):
    def __init__(self, mol: Molecule, cuda: bool = False):
        """Define a generic kernel to test the auto diff features."""
        super().__init__(mol, cuda)
        eps = 1e-4
        self.weight = nn.Parameter(eps * torch.rand(self.nelec, self.nelec)).to(
            self.device
        )

    def _backflow_kernel(self, ree: torch.Tensor) -> torch.Tensor:
        """Computes the backflow kernel:

        .. math:
            \\eta(r_{ij}) = w_{ij} r_{ij}^2

        Args:
            r (torch.tensor): e-e distance Nbatch x Nelec x Nelec

        Returns:
            torch.tensor : f(r) Nbatch x Nelec x Nelec
        """
        return self.weight * ree * ree
