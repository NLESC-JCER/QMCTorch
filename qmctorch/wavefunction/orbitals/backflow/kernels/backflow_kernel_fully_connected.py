import torch
from torch import nn
from torch.autograd import grad, Variable
from .backflow_kernel_base import BackFlowKernelBase


class BackFlowKernelFullyConnected(BackFlowKernelBase):

    def __init__(self, mol, cuda):
        """Compute the back flow kernel, i.e. the function
        f(rij) where rij is the distance between electron i and j
        This kernel is used in the backflow transformation
        .. math:
            q_i = r_i + \\sum_{j\\neq i} f(r_{ij}) (r_i-r_j)
        """
        super().__init__(mol, cuda)
        self.fc1 = nn.Linear(1, 16, bias=False)
        self.fc2 = nn.Linear(16, 1, bias=False)
        self.nl_func = torch.nn.Sigmoid()

        eps = 1E-0
        self.fc1.weight.data *= eps
        self.fc2.weight.data *= eps

    def _backflow_kernel(self, ree):
        """Computes the kernel via autodiff

        Args:
            ree ([type]): [description]

        Returns:
            [type]: [description]
        """
        original_shape = ree.shape
        x = ree.reshape(-1, 1)

        x = self.fc1(x)
        x = self.nl_func(x)
        x = self.fc2(x)
        x = self.nl_func(x)
        x = x.reshape(*original_shape)

        return x
