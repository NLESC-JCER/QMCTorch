import torch
from torch import nn
from .backflow_kernel_base import BackFlowKernelBase
from .....scf import Molecule


class BackFlowKernelPowerSum(BackFlowKernelBase):
    def __init__(self, mol: Molecule, cuda: bool, order: int = 2):
        """Compute the back flow kernel, i.e. the function
        f(rij) where rij is the distance between electron i and j
        This kernel is used in the backflow transformation
        .. math:
            q_i = r_i + \\sum_{j\\neq i} f(r_{ij}) (r_i-r_j)
        """
        super().__init__(mol, cuda)
        self.order = order
        self.fc = nn.Linear(order, 1, bias=False)
        self.fc.weight.data *= 0.0
        self.fc.weight.data[0, 0] = 1e-4

    def _backflow_kernel(self, ree: torch.Tensor) -> torch.Tensor:
        """Computes the kernel via autodiff

        Args:
            ree ([type]): [description]

        Returns:
            [type]: [description]
        """

        original_shape = ree.shape
        x = ree.reshape(-1, 1)
        x = x.repeat(1, self.order)
        x = x.cumprod(dim=-1)
        x = self.fc(x)
        x = x.reshape(*original_shape)

        return x
