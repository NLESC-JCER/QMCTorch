import torch
from torch import nn

from .....utils import register_extra_attributes
from .jastrow_kernel_electron_nuclei_base import JastrowKernelElectronNucleiBase


class QuadraticPadeJastrowKernel(JastrowKernelElectronNucleiBase):
    def __init__(
        self,
        nup: int,
        ndown: int,
        atomic_pos: torch.Tensor,
        cuda: bool,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
    ) -> None:
        """
        Initializes the Quadratic Pade-Jastrow kernel with the given parameters.

        Args:
            nup (int): Number of spin-up electrons.
            ndown (int): Number of spin-down electrons.
            atomic_pos (torch.Tensor): Tensor containing the atomic positions.
            cuda (bool): Flag to indicate whether to use CUDA for computations.
            a (float, optional): Initial value for the rweight parameter. Defaults to 1.0.
            b (float, optional): Initial value for the r2weight parameter. Defaults to 1.0.
            c (float, optional): Initial value for the weight parameter. Defaults to 1.0.
        """

        super().__init__(nup, ndown, atomic_pos, cuda)
        self.rweight = nn.Parameter(torch.as_tensor([a]), requires_grad=True).to(
            self.device
        )
        self.r2weight = nn.Parameter(torch.as_tensor([b]), requires_grad=True).to(
            self.device
        )
        self.weight = nn.Parameter(torch.as_tensor([c]), requires_grad=True).to(
            self.device
        )
        register_extra_attributes(self, ["weight", "r2weight", "rweight"])
        self.requires_autograd = True

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Computes the Quadratic Pade-Jastrow kernel.

        .. math::
            J(r) = \\frac{a r + b r^2}{1 + c r}

        Args:
            r (torch.Tensor): Tensor containing the e-n distances.

        Returns:
            torch.Tensor: Tensor containing the computed Jastrow kernel.
        """
        return (self.rweight * r + self.r2weight * r**2) / (1.0 + self.weight * r)
