import torch
from torch import nn
from .backflow_kernel_base import BackFlowKernelBase
from .....scf import Molecule


class BackFlowKernelAutoInverse(BackFlowKernelBase):
    def __init__(self, mol: Molecule, cuda: bool, weight: float = 0.0) -> None:
        """Compute the back flow kernel, i.e. the function
        f(rij) where rij is the distance between electron i and j
        This kernel is used in the backflow transformation
        .. math:
            q_i = r_i + \\sum_{j\\neq i} f(r_{ij}) (r_i-r_j)
        """
        super().__init__(mol, cuda)
        self.weight = nn.Parameter(torch.as_tensor([weight]))

    def _backflow_kernel(self, ree: torch.Tensor) -> torch.Tensor:
        """Computes the kernel via autodiff

        Args:
            ree ([type]): [description]

        Returns:
            [type]: [description]
        """
        eye = torch.eye(self.nelec, self.nelec).to(self.device)
        mask = torch.ones_like(ree) - eye
        return self.weight * mask * (1.0 / (ree + eye) - eye)
