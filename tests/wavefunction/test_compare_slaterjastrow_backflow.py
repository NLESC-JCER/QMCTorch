import numpy as np
import torch
import unittest

from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow

from qmctorch.wavefunction.jastrows.elec_elec.jastrow_factor_electron_electron import (
    JastrowFactorElectronElectron,
)
from qmctorch.wavefunction.jastrows.elec_elec.kernels import PadeJastrowKernel

from qmctorch.wavefunction.orbitals.backflow.backflow_transformation import (
    BackFlowTransformation,
)
from qmctorch.wavefunction.orbitals.backflow.kernels.backflow_kernel_inverse import (
    BackFlowKernelInverse,
)

from qmctorch.utils import set_torch_double_precision

set_torch_double_precision()


class TestCompareSlaterJastrowBackFlow(unittest.TestCase):
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
        jastrow = JastrowFactorElectronElectron(
            mol,
            PadeJastrowKernel,
        )

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

        self.wf.ao.backflow_trans.backflow_kernel.weight.data *= 0.0

        self.wf_ref = SlaterJastrow(
            mol,
            kinetic="jacobi",
            include_all_mo=True,
            configs="single_double(2,2)",
            jastrow=jastrow,
            backflow=None,
        )

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight
        self.wf_ref.fc.weight.data = self.random_fc_weight

        self.nbatch = 5
        self.pos = torch.Tensor(np.random.rand(self.nbatch, self.wf.nelec * 3))
        self.pos.requires_grad = True

    def test_forward(self):
        """Check that backflow give same results as normal SlaterJastrow."""
        wf_val = self.wf(self.pos)
        wf_val_ref = self.wf_ref(self.pos)

        assert torch.allclose(wf_val, wf_val_ref)

    def test_jacobian_mo(self):
        """Check that backflow give same results as normal SlaterJastrow."""

        dmo = self.wf.pos2mo(self.pos, derivative=1)
        dmo_ref = self.wf_ref.pos2mo(self.pos, derivative=1)
        assert torch.allclose(dmo.sum(0), dmo_ref)

    def test_hess_mo(self):
        """Check that backflow give same results as normal SlaterJastrow."""
        d2ao = self.wf.ao(self.pos, derivative=2, sum_hess=False)
        d2val = self.wf.ao2mo(d2ao)

        d2ao_ref = self.wf_ref.ao(self.pos, derivative=2, sum_hess=True)
        d2val_ref = self.wf_ref.ao2mo(d2ao_ref)
        assert torch.allclose(d2val_ref, d2val.sum(0))

    def test_local_energy(self):
        self.wf.kinetic_energy = self.wf.kinetic_energy_jacobi_backflow
        eloc_jac = self.wf.local_energy(self.pos)

        self.wf_ref.kinetic_energy = self.wf_ref.kinetic_energy_jacobi
        eloc_jac_ref = self.wf_ref.local_energy(self.pos)

        assert torch.allclose(eloc_jac_ref.data, eloc_jac.data, rtol=1e-4, atol=1e-4)

    def test_kinetic_energy(self):
        ejac_ref = self.wf_ref.kinetic_energy_jacobi(self.pos)
        ejac = self.wf.kinetic_energy_jacobi(self.pos)

        assert torch.allclose(ejac_ref.data, ejac.data, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    # t = TestCompareSlaterJastrowBackFlow()
    # t.setUp()
    # t.test_jacobian_mo()
    # t.test_hess_mo()
    # t.test_kinetic_energy()
    # t.test_local_energy()
    unittest.main()
