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


class BackFlowKernelAutoInverse(BackFlowKernelBase):

    def __init__(self, mol, cuda, order=2):
        """Compute the back flow kernel, i.e. the function
        f(rij) where rij is the distance between electron i and j
        This kernel is used in the backflow transformation
        .. math:
            q_i = r_i + \\sum_{j\\neq i} f(r_{ij}) (r_i-r_j)
        """
        super().__init__(mol, cuda)
        self.order = order
        self.fc = nn.Linear(order, 1, bias=False)
        self.fc.weight.data *= 0.
        self.fc.weight.data[0, 0] = 1.

        self.weight = nn.Parameter(
            torch.as_tensor([1E-3]))

    def _backflow_kernel(self, ree):
        """Computes the kernel via autodiff

        Args:
            ree ([type]): [description]

        Returns:
            [type]: [description]
        """
        eye = torch.eye(self.nelec, self.nelec).to(self.device)
        mask = torch.ones_like(ree) - eye
        return self.weight * mask * (1./(ree+eye) - eye)


class BackFlowKernelPowerSum(BackFlowKernelBase):

    def __init__(self, mol, cuda, order=2):
        """Compute the back flow kernel, i.e. the function
        f(rij) where rij is the distance between electron i and j
        This kernel is used in the backflow transformation
        .. math:
            q_i = r_i + \\sum_{j\\neq i} f(r_{ij}) (r_i-r_j)
        """
        super().__init__(mol, cuda)
        self.order = order
        self.fc = nn.Linear(order, 1, bias=False)
        self.fc.weight.data *= 0.
        self.fc.weight.data[0, 0] = 1E-4

    def _backflow_kernel(self, ree):
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


class GenericBackFlowKernel(BackFlowKernelBase):

    def __init__(self, mol, cuda=False):
        """Define a generic kernel to test the auto diff features."""
        super().__init__(mol, cuda)
        eps = 1E-4
        self.weight = nn.Parameter(
            eps * torch.rand(self.nelec, self.nelec)).to(self.device)

    def _backflow_kernel(self, ree):
        """Computes the backflow kernel:

        .. math:
            \\eta(r_{ij}) = w_{ij} r_{ij}^2

        Args:
            r (torch.tensor): e-e distance Nbatch x Nelec x Nelec

        Returns:
            torch.tensor : f(r) Nbatch x Nelec x Nelec
        """
        return self.weight * ree * ree
