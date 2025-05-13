import unittest
import numpy as np
import torch


from .base_elec_elec_jastrow_test import BaseTestJastrow
from types import SimpleNamespace
from qmctorch.wavefunction.jastrows.elec_elec.jastrow_factor_electron_electron import (
    JastrowFactorElectronElectron,
)
from qmctorch.wavefunction.jastrows.elec_elec.kernels.pade_jastrow_polynomial_kernel import (
    PadeJastrowPolynomialKernel,
)
from qmctorch.utils import set_torch_double_precision

set_torch_double_precision()


class TestScaledPadeJastrowPolynom(BaseTestJastrow.ElecElecJastrowBaseTest):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

        mol = SimpleNamespace(nup=4, ndown=4)
        self.nelec = mol.nup + mol.ndown

        self.jastrow = JastrowFactorElectronElectron(
            mol,
            PadeJastrowPolynomialKernel,
            kernel_kwargs={
                "order": 5,
                "weight_a": 0.1 * torch.ones(5),
                "weight_b": 0.1 * torch.ones(5),
            },
            scale=True,
        )
        self.nbatch = 10

        self.pos = torch.rand(self.nbatch, self.nelec * 3)
        self.pos.requires_grad = True


if __name__ == "__main__":
    unittest.main()
