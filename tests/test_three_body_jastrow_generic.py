import unittest

import numpy as np
import torch
from torch.autograd import Variable, grad, gradcheck
from qmctorch.wavefunction.jastrows.three_body_jastrow_generic import ThreeBodyJastrowFactorGeneric

torch.set_default_tensor_type(torch.DoubleTensor)


class FullyConnected(torch.nn.Module):
    def __init__(self):
        """Defines a fully connected jastrow factors."""

        super(FullyConnected, self).__init__()

        self.fc1 = torch.nn.Linear(3, 16, bias=True)
        self.fc2 = torch.nn.Linear(16, 8, bias=True)
        self.fc3 = torch.nn.Linear(8, 1, bias=True)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        # self.fc1.weight.data.fill_(1E-3)
        # self.fc2.weight.data.fill_(1E-3)
        # self.fc3.weight.data.fill_(1E-3)

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
        self.atoms = torch.rand(4, 3)
        self.jastrow = ThreeBodyJastrowFactorGeneric(
            self.nup, self.ndown, self.atoms, FullyConnected, False)
        self.nbatch = 5

        self.pos = torch.rand(self.nbatch, self.nelec * 3)
        self.pos.requires_grad = True

    def test_grad_distance(self):

        r = self.jastrow.elel_dist(self.pos)
        dr = self.jastrow.elel_dist(self.pos, derivative=1)
        dr_grad = grad(
            r,
            self.pos,
            grad_outputs=torch.ones_like(r))[0]

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
    # unittest.main()
    t = TestThreeBodyGeneric()
    t.setUp()
    t.test_grad_distance()
    t.test_grad_jastrow()
    # nup, ndown = 4, 4
    # nelec = nup + ndown
    # atoms = torch.rand(4, 3)
    # jastrow = ThreeBodyJastrowFactorGeneric(
    #     nup, ndown, atoms, FullyConnected, False)
    # nbatch = 5

    # pos = torch.rand(nbatch, nelec * 3)
    # pos.requires_grad = True

    # ree = jastrow.extract_tri_up(jastrow.elel_dist(pos))
    # ren = jastrow.extract_elec_nuc_dist(jastrow.elnu_dist(pos))
    # r = jastrow.assemble_dist(pos)

    # val = jastrow.jastrow_function(r)
    # gval = jastrow._grads(val, r)

    # dr = jastrow.assemble_dist_deriv(pos)
