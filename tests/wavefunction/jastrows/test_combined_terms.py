import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch.autograd import grad, gradcheck

from qmctorch.wavefunction.jastrows.combine_jastrow import (
    CombineJastrow,
)
from qmctorch.wavefunction.jastrows.elec_elec import (
    JastrowFactor as JastrowFactorElecElec,
    PadeJastrowKernel as ElecElecKernel,
)

from qmctorch.wavefunction.jastrows.elec_nuclei import (
    JastrowFactor as JastrowFactorElecNuclei,
    PadeJastrowKernel as ElecNucleiKernel,
)

from qmctorch.wavefunction.jastrows.elec_elec_nuclei import (
    BoysHandyJastrowKernel as ElecElecNucleiKernel,
    JastrowFactor as JastrowFactorElecElecNuc,
)
from qmctorch.utils import set_torch_double_precision
from qmctorch.utils.torch_utils import diagonal_hessian as hess

set_torch_double_precision()


class TestJastrowCombinedTerms(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

        self.nup, self.ndown = 4, 4
        self.nelec = self.nup + self.ndown
        self.atoms = np.random.rand(4, 3)

        self.mol = SimpleNamespace(
            nup=self.nup, ndown=self.ndown, atom_coords=self.atoms
        )

        jastrow_ee = JastrowFactorElecElec(
            self.mol, ElecElecKernel, kernel_kwargs={"w": 1.0}
        )

        jastrow_en = JastrowFactorElecNuclei(
            self.mol, ElecNucleiKernel, kernel_kwargs={"w": 1.0}
        )

        jastrow_een = JastrowFactorElecElecNuc(self.mol, ElecElecNucleiKernel)

        self.jastrow = CombineJastrow([jastrow_ee, jastrow_en, jastrow_een])

        self.nbatch = 5

        self.pos = torch.rand(self.nbatch, self.nelec * 3)
        self.pos.requires_grad = True

    def test_jastrow(self):
        _ = self.jastrow(self.pos)

    def test_grad_jastrow(self):
        val = self.jastrow(self.pos)
        dval = self.jastrow(self.pos, derivative=1)
        dval_grad = grad(val, self.pos, grad_outputs=torch.ones_like(val))[0]

        dval_grad = dval_grad.view(self.nbatch, self.nelec, 3).sum(2)
        gradcheck(self.jastrow, self.pos)

        assert torch.allclose(dval, dval_grad)
        assert torch.allclose(dval.sum(), dval_grad.sum())

    def test_hess_jastrow(self):
        val = self.jastrow(self.pos)
        d2val_grad, _ = hess(val, self.pos)
        d2val = self.jastrow(self.pos, derivative=2)

        assert torch.allclose(d2val.sum(), d2val_grad.sum())
        assert torch.allclose(d2val, d2val_grad.view(self.nbatch, self.nelec, 3).sum(2))


if __name__ == "__main__":
    unittest.main()
    # t = TestJastrowCombinedTerms()
    # t.setUp()
    # t.test_hess_jastrow()
