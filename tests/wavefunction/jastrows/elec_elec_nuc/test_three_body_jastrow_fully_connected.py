import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch.autograd import grad
from qmctorch.wavefunction.jastrows.elec_elec_nuclei.jastrow_factor_electron_electron_nuclei import (
    JastrowFactorElectronElectronNuclei,
)
from qmctorch.wavefunction.jastrows.elec_elec_nuclei.kernels.fully_connected_jastrow_kernel import (
    FullyConnectedJastrowKernel,
)
from qmctorch.utils import set_torch_double_precision
from qmctorch.utils.torch_utils import diagonal_hessian as hess

set_torch_double_precision()


class TestThreeBodyFullyConnected(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

        self.nup, self.ndown = 4, 4
        self.nelec = self.nup + self.ndown
        self.natom = 4
        self.atoms = 0.1 * torch.rand(self.natom, 3)
        self.mol = SimpleNamespace(
            nup=self.nup, ndown=self.ndown, atom_coords=self.atoms
        )
        self.jastrow = JastrowFactorElectronElectronNuclei(
            self.mol, FullyConnectedJastrowKernel
        )
        self.nbatch = 5

        self.pos = 0.1 * torch.rand(self.nbatch, self.nelec * 3)
        self.pos.requires_grad = True

    def test_grad_elel_distance(self):
        r = self.jastrow.elel_dist(self.pos)
        dr = self.jastrow.elel_dist(self.pos, derivative=1)
        dr_grad = grad(r, self.pos, grad_outputs=torch.ones_like(r))[0]

        dr_grad = dr_grad.reshape(self.nbatch, self.nelec, 3)
        dr = dr.sum(-1).permute(0, 2, 1)

        assert torch.allclose(2 * dr, dr_grad, atol=1e-5)

    def test_grad_elnu_distance(self):
        r = self.jastrow.elnu_dist(self.pos)
        dr = self.jastrow.elnu_dist(self.pos, derivative=1)
        dr_grad = grad(r, self.pos, grad_outputs=torch.ones_like(r))[0]

        dr_grad = dr_grad.reshape(self.nbatch, self.nelec, 3)
        dr = dr.sum(-1).permute(0, 2, 1)

        assert torch.allclose(dr, dr_grad, atol=1e-5)

    def test_jacobian_jastrow(self):
        val = self.jastrow(self.pos)
        dval = self.jastrow(self.pos, derivative=1)
        dval_grad = grad(val, self.pos, grad_outputs=torch.ones_like(val))[0]

        dval_grad = dval_grad.view(self.nbatch, self.nelec, 3).sum(2)

        assert torch.allclose(dval.sum(), dval_grad.sum())
        assert torch.allclose(dval, dval_grad)

    def test_grad_jastrow(self):
        val = self.jastrow(self.pos)
        dval = self.jastrow(self.pos, derivative=1, sum_grad=False)
        dval_grad = grad(val, self.pos, grad_outputs=torch.ones_like(val))[0]

        dval_grad = dval_grad.view(self.nbatch, self.nelec, 3)

        assert torch.allclose(dval.sum(), dval_grad.sum())
        assert torch.allclose(dval.permute(0, 2, 1), dval_grad)

    def test_hess_jastrow(self):
        val = self.jastrow(self.pos)
        d2val_grad, _ = hess(val, self.pos)
        d2val_grad = d2val_grad.view(self.nbatch, self.nelec, 3).sum(2)
        d2val = self.jastrow(self.pos, derivative=2)
        print(d2val_grad)
        print(d2val)
        assert torch.allclose(d2val.sum(), d2val_grad.sum())
        assert torch.allclose(d2val, d2val_grad)


if __name__ == "__main__":
    unittest.main()
