import unittest

import numpy as np
import torch
from torch.autograd import Variable, grad, gradcheck
from qmctorch.wavefunction.jastrows.elec_nuclei.electron_nuclei_generic import ElectronNucleiGeneric

torch.set_default_tensor_type(torch.DoubleTensor)


class FullyConnectedJastrowElecNuc(torch.nn.Module):
    def __init__(self):
        """Defines a fully connected jastrow factors."""

        super(FullyConnectedJastrowElecNuc, self).__init__()

        self.fc1 = torch.nn.Linear(1, 16, bias=False)
        self.fc2 = torch.nn.Linear(16, 8, bias=False)
        self.fc3 = torch.nn.Linear(8, 1, bias=False)

        torch.nn.init.uniform_(self.fc1.weight)
        torch.nn.init.uniform_(self.fc2.weight)
        torch.nn.init.uniform_(self.fc2.weight)

        # self.fc1.weight.data *= 1E-3
        # self.fc2.weight.data *= 1E-3
        # self.fc3.weight.data *= 1E-3

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
        x = self.nl_func(x)
        x = self.fc2(x)
        x = self.nl_func(x)
        x = self.fc3(x)
        x = self.nl_func(x)

        # reshape to the original shape
        x = x.reshape(*original_shape)

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


class TestElectronNucleiGeneric(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)
        np.random.seed(0)

        self.nup, self.ndown = 4, 4
        self.nelec = self.nup + self.ndown
        self.atoms = torch.rand(4, 3)
        self.jastrow = ElectronNucleiGeneric(
            self.nup, self.ndown, self.atoms, FullyConnectedJastrowElecNuc, False)
        self.nbatch = 5

        self.pos = torch.rand(self.nbatch, self.nelec * 3)
        self.pos.requires_grad = True

    def test_grad_distance(self):

        r = self.jastrow.edist(self.pos)
        dr = self.jastrow.edist(self.pos, derivative=1)
        dr_grad = grad(
            r,
            self.pos,
            grad_outputs=torch.ones_like(r))[0]
        gradcheck(self.jastrow.edist, self.pos)

        assert(torch.allclose(dr.sum(), dr_grad.sum(), atol=1E-5))

    def test_grad_jastrow(self):

        val = self.jastrow(self.pos)
        dval = self.jastrow(self.pos, derivative=1)
        dval_grad = grad(
            val,
            self.pos,
            grad_outputs=torch.ones_like(val))[0]

        dval_grad = dval_grad.view(
            self.nbatch, self.nelec, 3).sum(2)

        assert torch.allclose(dval, dval_grad)
        assert(torch.allclose(dval.sum(), dval_grad.sum()))

    def test_hess_jastrow(self):

        val = self.jastrow(self.pos)
        d2val_grad = hess(val, self.pos)
        d2val = self.jastrow(self.pos, derivative=2)

        assert torch.allclose(d2val, d2val_grad.view(
            self.nbatch, self.nelec, 3).sum(2))
        assert(torch.allclose(d2val.sum(), d2val_grad.sum()))


if __name__ == "__main__":
    unittest.main()
    # nup, ndown = 4, 4
    # nelec = nup + ndown
    # atoms = torch.rand(4, 3)
    # jastrow = ElectronNucleiPadeJastrow(nup, ndown, atoms)
    # nbatch = 5

    # pos = torch.rand(nbatch, nelec * 3)
    # pos.requires_grad = True

    # jastrow.edist(pos, derivative=2)
