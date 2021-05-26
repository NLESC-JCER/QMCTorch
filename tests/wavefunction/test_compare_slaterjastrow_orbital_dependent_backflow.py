from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrowBackFlow, SlaterJastrow
from qmctorch.utils import set_torch_double_precision
from qmctorch.wavefunction.orbitals.backflow.kernels import BackFlowKernelInverse

from torch.autograd import grad, Variable

import numpy as np
import torch
import unittest

torch.set_default_tensor_type(torch.DoubleTensor)


def hess(out, pos):
    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos,
                 grad_outputs=z,
                 only_inputs=True,
                 create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[1]):

        tmp = grad(jacob[:, idim], pos,
                   grad_outputs=z,
                   only_inputs=True,
                   create_graph=True)[0]

        hess[:, idim] = tmp[:, idim]

    return hess


class TestCompareSlaterJastrowOrbitalDependentBackFlow(unittest.TestCase):

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

        self.wf = SlaterJastrowBackFlow(mol,
                                        kinetic='jacobi',
                                        include_all_mo=True,
                                        configs='single_double(2,2)',
                                        backflow_kernel=BackFlowKernelInverse,
                                        orbital_dependent_backflow=True)

        for ker in self.wf.ao.backflow_trans.backflow_kernel.orbital_dependent_kernel:
            ker.weight.data *= 0

        self.wf_ref = SlaterJastrow(mol,
                                    kinetic='jacobi',
                                    include_all_mo=True,
                                    configs='single_double(2,2)')

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight
        self.wf_ref.fc.weight.data = self.random_fc_weight

        self.nbatch = 5
        self.pos = torch.Tensor(np.random.rand(
            self.nbatch,  self.wf.nelec*3))
        self.pos.requires_grad = True

    def test_forward(self):
        """Check that backflow give same results as normal SlaterJastrow."""
        wf_val = self.wf(self.pos)
        wf_val_ref = self.wf_ref(self.pos)

        assert(torch.allclose(wf_val, wf_val_ref))

    def test_jacobian_mo(self):
        """Check that backflow give same results as normal SlaterJastrow."""

        dmo = self.wf.pos2mo(self.pos, derivative=1)
        dmo_ref = self.wf_ref.pos2mo(self.pos, derivative=1)
        assert(torch.allclose(dmo.sum(0), dmo_ref))

    def test_hess_mo(self):
        """Check that backflow give same results as normal SlaterJastrow."""
        d2ao = self.wf.ao(self.pos, derivative=2, sum_hess=False)
        d2val = self.wf.ao2mo(d2ao)

        d2ao_ref = self.wf_ref.ao(
            self.pos, derivative=2, sum_hess=True)
        d2val_ref = self.wf_ref.ao2mo(d2ao_ref)
        assert(torch.allclose(d2val_ref, d2val.sum(0)))

    def test_grad_wf(self):
        pass

    def test_local_energy(self):

        self.wf.kinetic_energy = self.wf.kinetic_energy_jacobi
        eloc_jac = self.wf.local_energy(self.pos)

        self.wf_ref.kinetic_energy = self.wf_ref.kinetic_energy_jacobi
        eloc_jac_ref = self.wf_ref.local_energy(self.pos)

        assert torch.allclose(
            eloc_jac_ref.data, eloc_jac.data, rtol=1E-4, atol=1E-4)

    def test_kinetic_energy(self):

        ejac_ref = self.wf_ref.kinetic_energy_jacobi(self.pos)
        ejac = self.wf.kinetic_energy_jacobi(self.pos)

        assert torch.allclose(
            ejac_ref.data, ejac.data, rtol=1E-4, atol=1E-4)


if __name__ == "__main__":
    t = TestCompareSlaterJastrowOrbitalDependentBackFlow()
    t.setUp()
    t.test_jacobian_mo()
    t.test_hess_mo()
    t.test_kinetic_energy()
    t.test_local_energy()
    # unittest.main()
