import unittest
from qmctorch.scf import Molecule
import numpy as np
import torch
from .base_test_cases import BaseTestCases
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.wavefunction.jastrows.elec_elec.jastrow_factor_electron_electron import (
    JastrowFactorElectronElectron,
)
from qmctorch.wavefunction.jastrows.elec_elec.kernels import PadeJastrowKernel
from qmctorch.utils import set_torch_double_precision


class TestSlaterJastrow(BaseTestCases.WaveFunctionBaseTest):
    def setUp(self):
        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        mol = Molecule(
            atom="Li 0 0 0; H 0 0 3.14",
            unit="bohr",
            calculator="pyscf",
            basis="sto-3g",
            redo_scf=True,
        )

        # define jastrow factor
        jastrow = JastrowFactorElectronElectron(
            mol, PadeJastrowKernel, orbital_dependent_kernel=True
        )

        self.wf = SlaterJastrow(
            mol,
            kinetic="auto",
            include_all_mo=False,
            configs="single_double(2,2)",
            jastrow=jastrow,
            backflow=None,
        )

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight
        self.nbatch = 11
        self.pos = torch.Tensor(np.random.rand(self.nbatch, self.wf.nelec * 3))
        self.pos.requires_grad = True

    def test_gradients_wf(self):
        pass

    def test_gradients_pdf(self):
        pass

    def test_kinetic_energy(self):
        pass

    def test_local_energy(self):
        pass


if __name__ == "__main__":
    unittest.main()
