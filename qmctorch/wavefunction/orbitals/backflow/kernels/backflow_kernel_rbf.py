import torch
from torch import nn
from torch.nn import functional as F

from .....scf import Molecule
from .backflow_kernel_base import BackFlowKernelBase


class BackFlowKernelRBF(BackFlowKernelBase):
    def __init__(self, mol: Molecule, cuda: bool = False, num_rbf: int = 10):
        """
        Initialize the RBF kernel

        Parameters
        ----------
        mol : Molecule
            Molecule object
        num_rbf : int
            Number of radial basis functions
        cuda : bool
            Whether to use CUDA or not

        Attributes
        ----------
        centers : nn.Parameter
            Centers of the radial basis functions
        sigma : nn.Parameter
            Widths of the radial basis functions
        weight : nn.Parameter
            Weights of the radial basis functions
        fc : nn.Linear
            Linear layer to compute the kernel
        bias : nn.Parameter
            Bias of the kernel
        """
        super().__init__(mol, cuda)
        self.num_rbf = num_rbf

        self.centers = nn.Parameter(torch.linspace(0, 10, num_rbf))
        self.centers.requires_grad = True

        self.sigma = nn.Parameter(0.25 * torch.ones(num_rbf))
        self.sigma.requires_grad = True

        self.weight = nn.Parameter(torch.Tensor(num_rbf, 1))
        self.weight.data.fill_(1.0)
        self.weight.requires_grad = False

        self.fc = nn.Linear(num_rbf, 1, bias=False)
        self.fc.weight.data = 1e-6 * torch.linspace(1, 0, num_rbf)
        self.bias = None

    def _gaussian_kernel(self, ree: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel

        Args:
            ree (torch.tensor): Nbatch x [Nelec * Nelec]

        Returns:
            torch.tensor: Nbatch x [Nelec * Nelec]
        """
        return torch.exp(-((ree - self.centers) ** 2) / self.sigma)

    def _gaussian_kernel_derivative(self, ree: torch.Tensor) -> torch.Tensor:
        """Compute the derivative of the RBF kernel

        Args:
            ree (torch.tensor): Nbatch x [Nelec * Nelec]

        Returns:
            torch.tensor: Nbatch x [Nelec * Nelec]
        """
        return -2 * (ree - self.centers) / self.sigma * self._gaussian_kernel(ree)

    def _gaussian_kernel_second_derivative(self, ree: torch.Tensor) -> torch.Tensor:
        """Compute the second derivative of the RBF kernel

        Args:
            ree (torch.tensor): Nbatch x [Nelec * Nelec]

        Returns:
            torch.tensor: Nbatch x [Nelec * Nelec]
        """
        kernel = self._gaussian_kernel(ree)
        derivative = self._gaussian_kernel_derivative(ree)
        return (
            -2 / self.sigma * kernel
            - 2 * (ree - self.centers) / self.sigma * derivative
        )

    def _backflow_kernel(self, ree: torch.Tensor) -> torch.Tensor:
        """Compute the kernel

        Args:
            ree (torch.tensor): Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: Nbatch x Nelec x Nelec
        """
        original_shape = ree.shape
        x = ree.reshape(-1, 1)
        x = F.linear(x, self.weight, self.bias)
        x = self._gaussian_kernel(x)
        x = self.fc(x)
        x = x.reshape(*original_shape)
        return x

    def _backflow_kernel_derivative(self, ree: torch.Tensor) -> torch.Tensor:
        """Compute the derivative of the kernel

        Args:
            ree (torch.tensor): Nbatch x Nelec x Nelec
        """
        original_shape = ree.shape
        x = ree.reshape(-1, 1)
        x = F.linear(x, self.weight, self.bias)
        x = self._gaussian_kernel_derivative(x)
        x = self.fc(x)
        x = x.reshape(*original_shape)
        return x

    def _backflow_kernel_second_derivative(self, ree: torch.Tensor) -> torch.Tensor:
        """Compute the second derivative of the kernel

        Args:
            ree (torch.tensor): Nbatch x Nelec x Nelec
        """
        original_shape = ree.shape
        x = ree.reshape(-1, 1)
        x = F.linear(x, self.weight, self.bias)
        x = self._gaussian_kernel_second_derivative(x)
        x = self.fc(x)
        x = x.reshape(*original_shape)
        return x
