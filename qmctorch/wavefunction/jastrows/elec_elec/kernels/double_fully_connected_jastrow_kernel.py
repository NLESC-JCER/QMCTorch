import torch
from torch import nn
from .jastrow_kernel_electron_electron_base import JastrowKernelElectronElectronBase


class SpinPairFullyConnectedJastrowKernel(JastrowKernelElectronElectronBase):
    def __init__(
        self,
        nup: int,
        ndown: int,
        cuda: bool,
        size1: int = 16,
        size2: int = 8,
        eps=1e-6,
        activation: torch.nn.Module = torch.nn.Sigmoid(),
    ) -> None:
        """Defines a fully connected jastrow factors with a separate fully connected layers for same and opposite spin

        Args:
            nup (int): Number of spin up electrons.
            ndown (int): Number of spin down electrons.
            cuda (bool): Whether to use the GPU or not.
            size1 (int, optional): Number of neurons in the first hidden layer. Defaults to 16.
            size2 (int, optional): Number of neurons in the second hidden layer. Defaults to 8.
            eps (float, optional): Small value for initialization. Defaults to 1E-6.
            activation (torch.nn.Module, optional): Activation function. Defaults to torch.nn.Sigmoid.
        """

        super().__init__(nup, ndown, cuda)

        self.fc1_same = nn.Linear(1, size1, bias=True)
        self.fc2_same = nn.Linear(size1, size2, bias=True)
        self.fc3_same = nn.Linear(size2, 1, bias=True)

        self.fc1_opp = nn.Linear(1, size1, bias=True)
        self.fc2_opp = nn.Linear(size1, size2, bias=True)
        self.fc3_opp = nn.Linear(size2, 1, bias=True)

        self.fc1_same.weight.data *= eps
        self.fc2_same.weight.data *= eps
        self.fc3_same.weight.data *= eps

        self.fc1_opp.weight.data *= eps
        self.fc2_opp.weight.data *= eps
        self.fc3_opp.weight.data *= eps

        self.fc1_same.bias.data *= eps
        self.fc2_same.bias.data *= eps
        self.fc3_same.bias.data *= eps

        self.fc1_opp.bias.data *= eps
        self.fc2_opp.bias.data *= eps
        self.fc3_opp.bias.data *= eps

        self.nl_func = activation

        self.get_idx_pair()

        self.requires_autograd = True

    def get_idx_pair(self) -> None:
        """
        Generate the indices of the same spin and opposite spin pairs.

        The Jastrow factor is applied on all pair of electrons. To apply the
        same spin Jastrow kernel or the opposite spin Jastrow kernel, it is
        necessary to know the indices of the same spin and opposite spin pairs.
        This function generate the indices of the same spin and opposite spin
        pairs.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        nelec = self.nup + self.ndown

        same_idx_pair = []
        opp_idx_pair = []
        ipair = 0
        for i in range(nelec - 1):
            ispin = 0 if i < self.nup else 1
            for j in range(i + 1, nelec):
                jspin = 0 if j < self.nup else 1

                if ispin == jspin:
                    same_idx_pair.append(ipair)
                else:
                    opp_idx_pair.append(ipair)

                ipair += 1

        self.same_idx_pair = torch.as_tensor(same_idx_pair).to(self.device)
        self.opp_idx_pair = torch.as_tensor(opp_idx_pair).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the values of the kernel for same spin and opposite spin pairs.

        Args:
            x (torch.tensor): e-e distance Nbatch, Nele_pairs

        Returns:
            torch.tensor: values of the kernel
        """
        out = torch.zeros_like(x)
        if len(self.same_idx_pair) > 0:
            out[:, self.same_idx_pair] = self._fsame(x[:, self.same_idx_pair])
        if len(self.opp_idx_pair) > 0:
            out[:, self.opp_idx_pair] = self._fopp(x[:, self.opp_idx_pair])
        return out

    def _fsame(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the kernel values of same spin pairs

        Args:
            x (torch.tensor): e-e distance Nbatch, Nelec_pairs

        Returns:
            torch.tensor: values of the f_ij
        """
        nbatch, npairs = x.shape

        # reshape the input so that all elements are considered
        # independently of each other
        x = x.reshape(-1, 1)

        x = self.fc1_same(x)
        x = self.nl_func(x)
        x = self.fc2_same(x)
        x = self.nl_func(x)
        x = self.fc3_same(x)
        x = self.nl_func(x)

        # reshape to the original shape
        x = x.reshape(nbatch, npairs)

        return x

    def _fopp(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the kernel values of opposite spin pairs

        Args:
            x (torch.tensor): e-e distance Nbatch, Nelec_pairs

        Returns:
            torch.tensor: values of the f_ij
        """
        nbatch, npairs = x.shape

        # reshape the input so that all elements are considered
        # independently of each other
        x = x.reshape(-1, 1)

        x = self.fc1_opp(x)
        x = self.nl_func(x)
        x = self.fc2_opp(x)
        x = self.nl_func(x)
        x = self.fc3_opp(x)
        x = self.nl_func(x)

        # reshape to the original shape
        x = x.reshape(nbatch, npairs)

        return x
