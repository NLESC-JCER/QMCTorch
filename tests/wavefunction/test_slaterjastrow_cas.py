import unittest
import numpy as np
import torch

from .base_test_cases import BaseTestCases

from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow

from qmctorch.wavefunction.jastrows.elec_elec.jastrow_factor_electron_electron import (
    JastrowFactorElectronElectron,
)
from qmctorch.wavefunction.jastrows.elec_elec.kernels import PadeJastrowKernel
from qmctorch.utils import set_torch_double_precision


set_torch_double_precision()


class TestSlaterJastrowCAS(BaseTestCases.WaveFunctionBaseTest):
    def setUp(self):
        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        mol = Molecule(
            atom="Li 0 0 0; H 0 0 1.",
            unit="bohr",
            calculator="pyscf",
            basis="sto-3g",
            redo_scf=True,
        )

        # define jastrow factor
        jastrow = JastrowFactorElectronElectron(mol, PadeJastrowKernel)

        self.wf = SlaterJastrow(
            mol,
            kinetic="auto",
            include_all_mo=True,
            configs="cas(2,2)",
            jastrow=jastrow,
        )

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight

        self.nbatch = 10
        self.pos = torch.Tensor(np.random.rand(self.nbatch, mol.nelec * 3))
        self.pos.requires_grad = True


if __name__ == "__main__":
    unittest.main()
