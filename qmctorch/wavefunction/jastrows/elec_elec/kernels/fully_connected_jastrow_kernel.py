import torch
from torch import nn
from .jastrow_kernel_electron_electron_base import JastrowKernelElectronElectronBase


class FullyConnectedJastrowKernel(JastrowKernelElectronElectronBase):
    def __init__(
        self,
        nup: int,
        ndown: int,
        cuda: bool,
        size1: int = 16,
        size2: int = 8,
        eps: float = 1e-6,
        activation: torch.nn.Module = torch.nn.Sigmoid(),
        include_cusp_weight: bool = True,
    ) -> None:
        """Defines a fully connected jastrow factors.

        Args:
            nup (int): Number of spin up electrons.
            ndown (int): Number of spin down electrons.
            cuda (bool): Whether to use the GPU or not.
            size1 (int, optional): Number of neurons in the first hidden layer. Defaults to 16.
            size2 (int, optional): Number of neurons in the second hidden layer. Defaults to 8.
            eps (float, optional): Small value for initialization. Defaults to 1E-6.
            activation (torch.nn.Module, optional): Activation function. Defaults to torch.nn.Sigmoid.
            include_cusp_weight (bool, optional): Whether to include the cusp weights or not. Defaults to True.
        """

        super().__init__(nup, ndown, cuda)

        self.fc1 = nn.Linear(1, size1, bias=False)
        self.fc2 = nn.Linear(size1, size2, bias=False)
        self.fc3 = nn.Linear(size2, 1, bias=False)
        self.var_cusp_weight = nn.Parameter(torch.as_tensor([0.0, 0.0]))
        self.fc1.weight.data *= eps
        self.fc2.weight.data *= eps
        self.fc3.weight.data *= eps

        self.nl_func = activation

        self.get_idx_pair()
        self.include_cusp_weight = include_cusp_weight

        self.requires_autograd = True

    def get_idx_pair(self) -> None:
        """
        Generate the indices of the same spin and opposite spin pairs.

        The Jastrow factor is applied on all pair of electrons. To apply the
        same spin Jastrow kernel or the opposite spin Jastrow kernel, it is
        necessary to know the indices of the same spin and opposite spin pairs.
        This function generate the indices of the same spin and opposite spin
        pairs.

        Returns
        -------
        None
        """
        nelec = self.nup + self.ndown

        idx_pair = []
        for i in range(nelec - 1):
            ispin = 0 if i < self.nup else 1
            for j in range(i + 1, nelec):
                jspin = 0 if j < self.nup else 1

                if ispin == jspin:
                    idx_pair.append(0)
                else:
                    idx_pair.append(1)
        self.idx_pair = torch.as_tensor(idx_pair).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the kernel values

        Args:
            x (torch.tensor): e-e distance Nbatch, Nele_pairs

        Returns:
            torch.tensor: values of the f_ij
        """
        nbatch, npairs = x.shape

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
        x = x.reshape(nbatch, npairs)

        # add the cusp weight
        if self.include_cusp_weight:
            x = x + self.var_cusp_weight[self.idx_pair]
        return x
