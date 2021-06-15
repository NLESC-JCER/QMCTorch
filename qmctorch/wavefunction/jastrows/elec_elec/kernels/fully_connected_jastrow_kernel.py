import torch
from torch import nn
import numpy as np
from .jastrow_kernel_electron_electron_base import JastrowKernelElectronElectronBase


class FullyConnectedJastrowKernel(JastrowKernelElectronElectronBase):

    def __init__(self,  nup, ndown, cuda,
                 size1=16, size2=8,
                 activation=torch.nn.Sigmoid(),
                 include_cusp_weight=True):
        """Defines a fully connected jastrow factors."""

        super().__init__(nup, ndown, cuda)

        self.cusp_weights = None

        self.fc1 = nn.Linear(1, size1, bias=False)
        self.fc2 = nn.Linear(size1, size2, bias=False)
        self.fc3 = nn.Linear(size2, 1, bias=False)

        eps = 1E-6
        self.fc1.weight.data *= eps
        self.fc2.weight.data *= eps
        self.fc3.weight.data *= eps

        self.nl_func = activation
        #self.nl_func = lambda x:  x

        self.prefac = torch.rand(1)

        self.cusp_weights = self.get_static_weight()
        self.requires_autograd = True
        self.get_var_weight()

        self.include_cusp_weight = include_cusp_weight

    def get_var_weight(self):
        """define the variational weight."""

        nelec = self.nup + self.ndown

        self.var_cusp_weight = nn.Parameter(
            torch.as_tensor([0., 0.]))

        idx_pair = []
        for i in range(nelec-1):
            ispin = 0 if i < self.nup else 1
            for j in range(i+1, nelec):
                jspin = 0 if j < self.nup else 1

                if ispin == jspin:
                    idx_pair.append(0)
                else:
                    idx_pair.append(1)
        self.idx_pair = torch.as_tensor(idx_pair).to(self.device)

    def get_static_weight(self):
        """Get the matrix of static weights

        Returns:
            torch.tensor: static weight (0.5 (0.25) for parallel(anti) spins
        """

        bup = torch.cat((0.25 * torch.ones(self.nup, self.nup), 0.5 *
                         torch.ones(self.nup, self.ndown)), dim=1)

        bdown = torch.cat((0.5 * torch.ones(self.ndown, self.nup), 0.25 *
                           torch.ones(self.ndown, self.ndown)), dim=1)

        static_weight = torch.cat((bup, bdown), dim=0).to(self.device)

        mask_tri_up = torch.triu(torch.ones_like(
            static_weight), diagonal=1).type(torch.BoolTensor).to(self.device)
        static_weight = static_weight.masked_select(mask_tri_up)

        return static_weight

    def forward(self, x):
        """Compute the kernel values

        Args:
            x (torch.tensor): e-e distance Nbatch, Nele_pairs

        Returns:
            torch.tensor: values of the f_ij
        """
        nbatch, npairs = x.shape
        # w = (x*self.cusp_weights)

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
