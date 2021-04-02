import unittest

import numpy as np
import torch
from torch.autograd import Variable, grad, gradcheck
from qmctorch.wavefunction.jastrows.elec_elec_nuclei.electron_electron_nuclei_generic import ElectronElectronNucleiGeneric

torch.set_default_tensor_type(torch.DoubleTensor)


class DummyJastrow(torch.nn.Module):
    def __init__(self):
        """Defines a fully connected jastrow factors."""

        super(DummyJastrow, self).__init__()

    def forward(self, x):
        out_shape = list(x.shape)[:-1] + [1]
        x = x.reshape(-1, 3)
        x = (0.5*torch.cos(x**4)).sum(-1)
        return x.reshape(*out_shape)


class FullyConnected(torch.nn.Module):
    def __init__(self):
        """Defines a fully connected jastrow factors."""

        super(FullyConnected, self).__init__()

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


class TestThreeBodyGeneric(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)
        np.random.seed(0)

        self.nup, self.ndown = 4, 4
        self.nelec = self.nup + self.ndown
        self.natom = 4
        self.atoms = 0.1*torch.rand(self.natom, 3)
        self.jastrow = ElectronElectronNucleiGeneric(
            self.nup, self.ndown, self.atoms, FullyConnected, False)
        self.nbatch = 5

        self.pos = 0.1*torch.rand(self.nbatch, self.nelec * 3)
        self.pos.requires_grad = True

    def test_grad_elel_distance(self):

        r = self.jastrow.elel_dist(self.pos)
        dr = self.jastrow.elel_dist(self.pos, derivative=1)
        dr_grad = grad(
            r,
            self.pos,
            grad_outputs=torch.ones_like(r))[0]

        dr_grad = dr_grad.reshape(self.nbatch, self.nelec, 3)
        dr = dr.sum(-1).permute(0, 2, 1)

        assert(torch.allclose(2*dr, dr_grad, atol=1E-5))

    def test_grad_elnu_distance(self):

        r = self.jastrow.elnu_dist(self.pos)
        dr = self.jastrow.elnu_dist(self.pos, derivative=1)
        dr_grad = grad(
            r,
            self.pos,
            grad_outputs=torch.ones_like(r))[0]

        dr_grad = dr_grad.reshape(self.nbatch, self.nelec, 3)
        dr = dr.sum(-1).permute(0, 2, 1)

        assert(torch.allclose(dr, dr_grad, atol=1E-5))

    def test_jacobian_jastrow(self):

        val = self.jastrow(self.pos)
        dval = self.jastrow(self.pos, derivative=1)
        dval_grad = grad(
            val,
            self.pos,
            grad_outputs=torch.ones_like(val))[0]

        dval_grad = dval_grad.view(
            self.nbatch, self.nelec, 3).sum(2)

        assert(torch.allclose(dval.sum(), dval_grad.sum()))
        assert torch.allclose(dval, dval_grad)

    def test_grad_jastrow(self):

        val = self.jastrow(self.pos)
        dval = self.jastrow(self.pos, derivative=1, jacobian=False)
        dval_grad = grad(
            val,
            self.pos,
            grad_outputs=torch.ones_like(val))[0]

        dval_grad = dval_grad.view(
            self.nbatch, self.nelec, 3)

        assert(torch.allclose(dval.sum(), dval_grad.sum()))
        assert torch.allclose(dval.permute(0, 2, 1), dval_grad)

    def test_hess_jastrow(self):

        val = self.jastrow(self.pos)
        d2val_grad = hess(val, self.pos).view(
            self.nbatch, self.nelec, 3).sum(2)
        d2val = self.jastrow(self.pos, derivative=2)
        print(d2val_grad)
        print(d2val)
        assert(torch.allclose(d2val.sum(), d2val_grad.sum()))
        assert torch.allclose(d2val, d2val_grad)


if __name__ == "__main__":
    unittest.main()
