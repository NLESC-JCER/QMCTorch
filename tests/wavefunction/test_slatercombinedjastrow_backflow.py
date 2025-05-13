import numpy as np
import torch
import unittest

from .base_test_cases import BaseTestCases

from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow

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

from qmctorch.wavefunction.orbitals.backflow.backflow_transformation import (
    BackFlowTransformation,
)
from qmctorch.wavefunction.orbitals.backflow.kernels.backflow_kernel_inverse import (
    BackFlowKernelInverse,
)

from qmctorch.utils import set_torch_double_precision

set_torch_double_precision()


class TestSlaterJastrowBackFlow(BaseTestCases.BackFlowWaveFunctionBaseTest):
    def setUp(self):
        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        mol = Molecule(
            atom="Li 0 0 0; H 0 0 3.015",
            unit="bohr",
            calculator="pyscf",
            basis="sto-3g",
            redo_scf=True,
        )

        # define jastrow factor
        jastrow_ee = JastrowFactorElecElec(
            mol, ElecElecKernel, kernel_kwargs={"w": 1.0}
        )

        jastrow_en = JastrowFactorElecNuclei(
            mol, ElecNucleiKernel, kernel_kwargs={"w": 1.0}
        )

        jastrow_een = JastrowFactorElecElecNuc(mol, ElecElecNucleiKernel)

        jastrow = CombineJastrow([jastrow_ee, jastrow_en, jastrow_een])

        # define backflow trans
        backflow = BackFlowTransformation(mol, BackFlowKernelInverse)

        self.wf = SlaterJastrow(
            mol,
            kinetic="jacobi",
            include_all_mo=True,
            configs="single_double(2,2)",
            jastrow=jastrow,
            backflow=backflow,
        )
        # needed to use the same tests base case for all wave function
        self.wf.kinetic_energy_jacobi = self.wf.kinetic_energy_jacobi_backflow

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight

        self.nbatch = 5
        self.pos = torch.Tensor(np.random.rand(self.nbatch, self.wf.nelec * 3))
        self.pos.requires_grad = True


if __name__ == "__main__":
    unittest.main()
    # t = TestSlaterJastrowBackFlow()
    # t.setUp()
    # t.test_antisymmetry()
    # t.test_hess_mo()
    # t.test_grad_mo()
    # t.test_kinetic_energy()
