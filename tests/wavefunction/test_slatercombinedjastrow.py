import unittest
import numpy as np
import torch

from .base_test_cases import BaseTestCases

from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.utils import set_torch_double_precision
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


set_torch_double_precision()


class TestSlaterCombinedJastrow(BaseTestCases.WaveFunctionBaseTest):
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

        jastrow_ee = JastrowFactorElecElec(
            mol, ElecElecKernel, kernel_kwargs={"w": 1.0}
        )

        jastrow_en = JastrowFactorElecNuclei(
            mol, ElecNucleiKernel, kernel_kwargs={"w": 1.0}
        )

        jastrow_een = JastrowFactorElecElecNuc(mol, ElecElecNucleiKernel)

        jastrow = CombineJastrow([jastrow_ee, jastrow_en, jastrow_een])

        self.wf = SlaterJastrow(
            mol,
            kinetic="auto",
            include_all_mo=False,
            configs="single_double(2,2)",
            jastrow=jastrow,
        )

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight
        self.nbatch = 11
        self.pos = torch.Tensor(np.random.rand(self.nbatch, self.wf.nelec * 3))
        self.pos.requires_grad = True


if __name__ == "__main__":
    unittest.main()
    # t = TestSlaterCombinedJastrow()
    # t.setUp()
    # t.test_antisymmetry()
    # t.test_kinetic_energy()
