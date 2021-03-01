import torch
from torch.autograd import grad


import unittest

import numpy as np
import torch
from torch.autograd import Variable, grad, gradcheck
from qmctorch.wavefunction.jastrows.generic_jastrow import GenericJastrow

torch.set_default_tensor_type(torch.DoubleTensor)


class FullyConnectedJastrow(torch.nn.Module):

    def __init__(self):
        """Defines a fully connected jastrow factors."""

        super(FullyConnectedJastrow, self).__init__()

        self.cusp_weights = None

        self.fc1 = torch.nn.Linear(1, 16, bias=False)
        self.fc2 = torch.nn.Linear(16, 8, bias=False)
        self.fc3 = torch.nn.Linear(8, 1, bias=False)

        self.fc1.weight.data.fill_(1E-3)
        self.fc2.weight.data.fill_(1E-3)
        self.fc3.weight.data.fill_(1E-3)

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

        # reshape the input so that all elements are considered
        # independently of each other
        x = x.reshape(-1, 1)

        x = self.fc1(x)
        x = self.nl_func(x)
        x = self.fc2(x)
        x = self.nl_func(x)
        x = self.fc3(x)
        x = self.nl_func(x)
        x = self.nl_func(x)

        # reshape to the original shape
        x = x.reshape(nbatch, npairs)

        # add the cusp weight
        x = x + self.cusp_weights

        return x


def hess(out, pos):
    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos,
                 grad_outputs=z,
                 only_inputs=True,
                 create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[1]):

        tmp = grad(jacob[:, idim], pos,
                   grad_outputs=z,
                   only_inputs=True,
                   create_graph=True)[0]

        hess[:, idim] = tmp[:, idim]

    return hess


class TestGenericJastrow(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)
        np.random.seed(0)

        self.nup, self.ndown = 4, 4
        self.nelec = self.nup + self.ndown
        self.jastrow = GenericJastrow(
            self.nup, self.ndown, FullyConnectedJastrow, False)
        self.nbatch = 5

        self.pos = 1E-1 * torch.rand(self.nbatch, self.nelec * 3)
        self.pos.requires_grad = True

    def test_jastrow(self):
        """simply checks that the values are not crashing."""
        val = self.jastrow(self.pos)

    def test_grad_jastrow(self):
        """Checks the values of the gradients."""
        val = self.jastrow(self.pos)
        dval = self.jastrow(self.pos, derivative=1, jacobian=False)

        dval_grad = grad(
            val,
            self.pos,
            grad_outputs=torch.ones_like(val))[0]

        dval_grad = dval_grad.reshape(
            self.nbatch, self.nelec, 3).permute(0, 2, 1)

        assert(torch.allclose(dval, dval_grad))

    def test_jacobian_jastrow(self):
        """Checks the values of the gradients."""
        val = self.jastrow(self.pos)
        dval = self.jastrow(self.pos, derivative=1)

        dval_grad = grad(
            val,
            self.pos,
            grad_outputs=torch.ones_like(val))[0]

        dval_grad = dval_grad.reshape(
            self.nbatch, self.nelec, 3).permute(0, 2, 1).sum(-2)

        assert torch.allclose(dval, dval_grad)

    def test_hess_jastrow(self):

        val = self.jastrow(self.pos)
        d2val = self.jastrow(self.pos, derivative=2)
        d2val_grad = hess(val, self.pos)
        # print(d2val)
        # print(d2val_grad.reshape(
        #     self.nbatch, self.nelec, 3).sum(2))
        assert torch.allclose(d2val, d2val_grad.reshape(
            self.nbatch, self.nelec, 3).sum(2))


if __name__ == "__main__":
    unittest.main()
    # t = TestGenericJastrow()
    # t.setUp()
    # t.test_jastrow()
    # t.test_grad_jastrow()
    # t.test_jacobian_jastrow()
    # t.test_hess_jastrow()
