import torch
from torch import nn
import numpy as np


class FullyConnectedJastrow(torch.nn.Module):

    def __init__(self):
        """Defines a fully connected jastrow factors."""

        super(FullyConnectedJastrow, self).__init__()

        self.cusp_weights = None

        self.fc1 = nn.Linear(1, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)
        self.fc3 = nn.Linear(8, 1, bias=False)

        eps = 1E-0
        self.fc1.weight.data *= eps
        self.fc2.weight.data *= eps
        self.fc3.weight.data *= eps

        self.nl_func = torch.nn.Sigmoid()
        # self.nl_func = lambda x:  x

        self.prefac = torch.rand(1)

    @staticmethod
    def get_cusp_weights(npairs):
        """Computes the cusp bias

        Args:
            npairs (int): number of elec pairs
        """
        nelec = int(0.5 * (-1 + np.sqrt(1+8*npairs)))
        weights = torch.zeros(npairs)

        spin = torch.ones(nelec)
        spin[:int(nelec/2)] = -1
        ip = 0
        for i1 in range(nelec):
            for i2 in range(i1, nelec):
                if spin[i1] == spin[i2]:
                    weights[ip] = 0.25
                else:
                    weights[ip] = 0.5
                ip += 1
        return weights

    def forward(self, x):
        """Compute the values of the individual f_ij=f(r_ij)

        Args:
            x (torch.tensor): e-e distance Nbatch, Nele_pairs

        Returns:
            torch.tensor: values of the f_ij
        """
        nbatch, npairs = x.shape

        if self.cusp_weights is None:
            self.cusp_weights = self.get_cusp_weights(npairs)
        w = (x*self.cusp_weights)

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
        x = x + w
        print(torch.exp(x[0].sum()))
        return x


class FullyConnectedJastrowElecNuc(torch.nn.Module):
    def __init__(self):
        """Defines a fully connected jastrow factors."""

        super(FullyConnectedJastrowElecNuc, self).__init__()

        self.fc1 = nn.Linear(1, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)
        self.fc3 = nn.Linear(8, 1, bias=False)

        self.fc1.weight.data.fill_(1E-3)
        self.fc2.weight.data.fill_(1E-3)
        self.fc3.weight.data.fill_(1E-3)

        self.nl_func = torch.nn.Sigmoid()

    def forward(self, x):
        """Compute the values of the individual f_ij=f(r_ij)

        Args:
            x (torch.tensor): e-e distance Nbatch, Nele_pairs

        Returns:
            torch.tensor: values of the f_ij
        """
        original_shape = x.shape

        # reshape the input so that all elements
        # are considered independently of each other
        x = x.reshape(-1, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.nl_func(x)

        # reshape to the original shape
        x = x.reshape(*original_shape)

        return x
