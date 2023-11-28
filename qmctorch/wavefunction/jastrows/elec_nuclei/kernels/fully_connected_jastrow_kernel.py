import torch
from torch import nn

from .jastrow_kernel_electron_nuclei_base import JastrowKernelElectronNucleiBase


class FullyConnectedJastrowKernel(JastrowKernelElectronNucleiBase):

    def __init__(self, nup, ndown, atomic_pos, cuda, w=1.):
        r"""Computes the Simple Pade-Jastrow factor

        .. math::
            J = \prod_{i<j} \exp(B_{ij}) \quad \quad \\text{with} \quad \quad
            B_{ij} = \\frac{w_0 r_{i,j}}{1 + w r_{i,j}}

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            atoms (torch.tensor): atomic positions of the atoms
            w (float, optional): Value of the variational parameter. Defaults to 1..
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super().__init__(nup, ndown, atomic_pos, cuda)

        self.fc1 = nn.Linear(1, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)
        self.fc3 = nn.Linear(8, 1, bias=False)

        self.nl_func = torch.nn.Sigmoid()
        self.requires_autograd = True

    def forward(self, x):
        """ Get the jastrow kernel.

        Args:
            x (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nnuc

        Returns:
            torch.tensor: matrix of the jastrow kernels
                          Nbatch x Nelec x Nnuc
        """
        original_shape = x.shape

        # reshape the input so that all elements are considered
        # independently of each other
        x = x.reshape(-1, 1)

        x = self.fc1(x)
        x = self.nl_func(x)
        x = self.fc2(x)
        x = self.nl_func(x)
        x = self.fc3(x)
        x = self.nl_func(x)

        # reshape to the original shape
        x = x.reshape(*original_shape)

        return x
