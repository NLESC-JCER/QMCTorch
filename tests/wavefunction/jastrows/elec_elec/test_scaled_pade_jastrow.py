import unittest
import numpy as np
import torch

from .base_elec_elec_jastrow_test import BaseTestJastrow

from types import SimpleNamespace
from qmctorch.wavefunction.jastrows.elec_elec.jastrow_factor_electron_electron import (
    JastrowFactorElectronElectron,
)
from qmctorch.wavefunction.jastrows.elec_elec.kernels.pade_jastrow_kernel import (
    PadeJastrowKernel,
)
from qmctorch.utils import set_torch_double_precision

set_torch_double_precision()


class TestScaledPadeJastrow(BaseTestJastrow.ElecElecJastrowBaseTest):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

        mol = SimpleNamespace(nup=2, ndown=2)
        self.nelec = mol.nup + mol.ndown

        self.jastrow = JastrowFactorElectronElectron(
            mol, PadeJastrowKernel, kernel_kwargs={"w": 0.1}, scale=True
        )
        self.nbatch = 5

        self.pos = torch.rand(self.nbatch, self.nelec * 3)
        self.pos.requires_grad = True


if __name__ == "__main__":
    unittest.main()
