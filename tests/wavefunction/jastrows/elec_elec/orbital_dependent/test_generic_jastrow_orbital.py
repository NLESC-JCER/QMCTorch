import torch
from torch.autograd import grad


import unittest

import numpy as np
import torch
from torch.autograd import Variable, grad

from qmctorch.wavefunction.jastrows.elec_elec.jastrow_factor_electron_electron import JastrowFactorElectronElectron
from qmctorch.wavefunction.jastrows.elec_elec.kernels.fully_connected_jastrow_kernel import FullyConnectedJastrowKernel
from qmctorch.utils import set_torch_double_precision

set_torch_double_precision()


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


class TestGenericJastrowOrbital(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)
        np.random.seed(0)

        self.nup, self.ndown = 2, 2
        self.nelec = self.nup + self.ndown
        self.nmo = 10
        self.jastrow = JastrowFactorElectronElectron(
            self.nup, self.ndown,
            FullyConnectedJastrowKernel,
            orbital_dependent_kernel=True,
            number_of_orbitals=self.nmo
        )
        self.nbatch = 11

        self.pos = 1E-1 * torch.rand(self.nbatch, self.nelec * 3)
        self.pos.requires_grad = True

    def test_jastrow(self):
        """simply checks that the values are not crashing."""
        val = self.jastrow(self.pos)

    def test_grad_jastrow(self):
        """Checks the values of the gradients."""
        val = self.jastrow(self.pos)
        dval = self.jastrow(self.pos, derivative=1, sum_grad=False)

        dval_grad = grad(
            val,
            self.pos,
            grad_outputs=torch.ones_like(val))[0]

        dval_grad = dval_grad.reshape(
            self.nbatch, self.nelec, 3).permute(0, 2, 1)

        # Warning : using grad on a model made out of ModuleList
        # automatically summ the values of the grad of the different
        # modules in the list !
        assert(torch.allclose(dval.sum(0), dval_grad))

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

        # Warning : using grad on a model made out of ModuleList
        # automatically summ the values of the grad of the different
        # modules in the list !
        assert torch.allclose(dval.sum(0), dval_grad)

    def test_hess_jastrow(self):

        val = self.jastrow(self.pos)
        d2val = self.jastrow(self.pos, derivative=2)
        d2val_grad = hess(val, self.pos)

        # Warning : using grad on a model made out of ModuleList
        # automatically summ the values of the grad of the different
        # modules in the list !
        assert torch.allclose(d2val.sum(0), d2val_grad.reshape(
            self.nbatch, self.nelec, 3).sum(2))


if __name__ == "__main__":
    unittest.main()

    # t = TestGenericJastrowOrbital()
    # t.setUp()
    # t.test_jastrow()
    # t.test_grad_jastrow()
    # t.test_jacobian_jastrow()
    # t.test_hess_jastrow()
