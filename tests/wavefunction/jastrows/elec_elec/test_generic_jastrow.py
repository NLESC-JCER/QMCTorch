import unittest
import numpy as np
import torch


from .base_elec_elec_jastrow_test import BaseTestJastrow

from types import SimpleNamespace
from qmctorch.wavefunction.jastrows.elec_elec.jastrow_factor_electron_electron import (
    JastrowFactorElectronElectron,
)
from qmctorch.wavefunction.jastrows.elec_elec.kernels.fully_connected_jastrow_kernel import (
    FullyConnectedJastrowKernel,
)
from qmctorch.utils import set_torch_double_precision

set_torch_double_precision()


class TestGenericJastrow(BaseTestJastrow.ElecElecJastrowBaseTest):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

        mol = SimpleNamespace(nup=4, ndown=4)
        self.nelec = mol.nup + mol.ndown

        self.jastrow = JastrowFactorElectronElectron(mol, FullyConnectedJastrowKernel)
        self.nbatch = 5

        self.pos = 1e-1 * torch.rand(self.nbatch, self.nelec * 3)
        self.pos.requires_grad = True


if __name__ == "__main__":
    unittest.main()
    # t = TestGenericJastrow()
    # t.setUp()
    # t.test_jastrow()
    # t.test_grad_jastrow()
    # t.test_jacobian_jastrow()
    # t.test_hess_jastrow()
