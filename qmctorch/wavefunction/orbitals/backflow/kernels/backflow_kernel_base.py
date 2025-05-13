import torch
from torch import nn
from .....scf import Molecule
from .....utils import gradients, hessian


class BackFlowKernelBase(nn.Module):
    def __init__(self, mol: Molecule, cuda: bool):
        """Compute the back flow kernel, i.e. the function
        f(rij) where rij is the distance between electron i and j
        This kernel is used in the backflow transformation
        .. math:
            q_i = r_i + \\sum_{j\\neq i} f(r_{ij}) (r_i-r_j)
        """
        super().__init__()
        self.nelec = mol.nelec
        self.cuda = cuda
        self.device = torch.device("cpu")
        if self.cuda:
            self.device = torch.device("cuda")

    def forward(self, ree: torch.Tensor, derivative: int = 0) -> torch.Tensor:
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
            raise ValueError("derivative of the kernel must be 0, 1 or 2")

    def _backflow_kernel(self, ree: torch.Tensor) -> torch.Tensor:
        """Computes the kernel via autodiff

        Args:
            ree ([type]): [description]

        Returns:
            [type]: [description]
        """
        raise NotImplementedError("Please implement the backflow kernel")

    def _backflow_kernel_derivative(self, ree: torch.Tensor) -> torch.Tensor:
        """Computes the first derivative of the kernel via autodiff

        Args:
            ree ([type]): [description]

        Returns:
            [type]: [description]
        """
        if ree.requires_grad == False:
            ree.requires_grad = True

        with torch.enable_grad():
            kernel_val = self._backflow_kernel(ree)

        return gradients(kernel_val, ree)

    def _backflow_kernel_second_derivative(self, ree: torch.Tensor) -> torch.Tensor:
        """Computes the second derivative of the kernel via autodiff

        Args:
            ree ([type]): [description]

        Returns:
            [type]: [description]
        """
        if ree.requires_grad == False:
            ree.requires_grad = True

        with torch.enable_grad():
            kernel_val = self._backflow_kernel(ree)
            hess_val, _ = hessian(kernel_val, ree)

        # if the kernel is linear, hval is None
        if hess_val is None:
            hess_val = torch.zeros_like(ree)

        return hess_val
