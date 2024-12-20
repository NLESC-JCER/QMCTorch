import unittest
import numpy as np
import torch

from .base_test_cases import BaseTestCases


from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow

from qmctorch.wavefunction.jastrows.elec_elec.jastrow_factor_electron_electron import (
    JastrowFactorElectronElectron,
)
from qmctorch.wavefunction.jastrows.elec_elec.kernels import FullyConnectedJastrowKernel


from qmctorch.utils import set_torch_double_precision


set_torch_double_precision()


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
        jastrow = JastrowFactorElectronElectron(mol, FullyConnectedJastrowKernel)

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


if __name__ == "__main__":
    unittest.main()
