import torch
from .jastrow_kernel_electron_electron_nuclei_base import JastrowKernelElectronElectronNucleiBase


class FullyConnectedJastrowKernel(JastrowKernelElectronElectronNucleiBase):

    def __init__(self, nup, ndown, atomic_pos, cuda):
        """Defines a fully connected jastrow factors."""

        super().__init__(nup, ndown, atomic_pos, cuda)

        self.fc1 = torch.nn.Linear(3, 9, bias=True)
        self.fc2 = torch.nn.Linear(9, 3, bias=True)
        self.fc3 = torch.nn.Linear(3, 1, bias=True)

        torch.nn.init.uniform_(self.fc1.weight)
        torch.nn.init.uniform_(self.fc2.weight)
        torch.nn.init.uniform_(self.fc2.weight)

        self.fc1.weight.data *= 1E-3
        self.fc2.weight.data *= 1E-3
        self.fc3.weight.data *= 1E-3

        self.nl_func = torch.nn.Sigmoid()

    def forward(self, x):
        """Compute the values of the individual f_ij=f(r_ij)

        Args:
            x (torch.tensor): e-e distance Nbatch, Nele_pairs

        Returns:
            torch.tensor: values of the f_ij
        """

        # reshape the input so that all elements
        # are considered independently of each other
        out_shape = list(x.shape)[:-1] + [1]
        x = x.reshape(-1, 3)

        x = self.fc1(x)
        x = 2 * self.nl_func(x)
        x = self.fc2(x)
        x = 2 * self.nl_func(x)
        x = self.fc3(x)
        x = 2 * self.nl_func(x)

        return x.reshape(*out_shape)
