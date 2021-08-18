

import numpy as np
import torch
import unittest

from .base_test_cases import BaseTestCases

from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow_unified import SlaterJastrowUnified as SlaterJastrow

from qmctorch.wavefunction.jastrows.elec_elec.jastrow_factor_electron_electron import JastrowFactorElectronElectron
from qmctorch.wavefunction.jastrows.elec_elec.kernels import PadeJastrowKernel

from qmctorch.wavefunction.orbitals.backflow.backflow_transformation import BackFlowTransformation
from qmctorch.wavefunction.orbitals.backflow.kernels.backflow_kernel_inverse import BackFlowKernelInverse

from qmctorch.utils import set_torch_double_precision


torch.set_default_tensor_type(torch.DoubleTensor)


class TestSlaterJastrowOrbitalDependentBackFlow(BaseTestCases.BackFlowWaveFunctionBaseTest):

    def setUp(self):

        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        mol = Molecule(
            atom='Li 0 0 0; H 0 0 3.015',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g',
            redo_scf=True)

        # define jastrow factor
        jastrow = JastrowFactorElectronElectron(
            mol, PadeJastrowKernel)

        # define backflow trans
        backflow = BackFlowTransformation(
            mol, BackFlowKernelInverse, orbital_dependent=True)

        self.wf = SlaterJastrow(mol,
                                kinetic='jacobi',
                                include_all_mo=True,
                                configs='single_double(2,2)',
                                jastrow=jastrow,
                                backflow=backflow)

        # change the weights
        for ker in self.wf.ao.backflow_trans.backflow_kernel.orbital_dependent_kernel:
            ker.weight.data[0] = torch.rand(1)

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight

        self.nbatch = 5
        self.pos = torch.Tensor(np.random.rand(
            self.nbatch,  self.wf.nelec*3))
        self.pos.requires_grad = True


if __name__ == "__main__":
    # t = TestSlaterJastrowOrbitalDependentBackFlow()
    # t.setUp()
    # t.test_antisymmetry()
    # t.test_hess_mo()
    # t.test_grad_mo()
    # t.test_kinetic_energy()
    unittest.main()
