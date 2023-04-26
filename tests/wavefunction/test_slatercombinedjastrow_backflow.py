from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterManyBodyJastrowBackflow
from qmctorch.utils import set_torch_double_precision
from qmctorch.wavefunction.jastrows.elec_nuclei.kernels import PadeJastrowKernel as PadeJastrowKernelElecNuc
from qmctorch.wavefunction.jastrows.elec_elec.kernels import PadeJastrowKernel as PadeJastrowKernelElecElec
from qmctorch.wavefunction.jastrows.elec_elec_nuclei.kernels import BoysHandyJastrowKernel, FullyConnectedJastrowKernel
from qmctorch.wavefunction.orbitals.backflow.kernels import BackFlowKernelInverse

from torch.autograd import grad, gradcheck, Variable

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


class TestSlaterCombinedJastrowBackflow(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        mol = Molecule(
            atom='Li 0 0 0; H 0 0 3.14',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g',
            redo_scf=True)

        self.wf = SlaterManyBodyJastrowBackflow(mol,
                                                kinetic='auto',
                                                include_all_mo=False,
                                                configs='single_double(2,2)',
                                                jastrow_kernel={
                                                    'ee': PadeJastrowKernelElecElec,
                                                    'en': PadeJastrowKernelElecNuc,
                                                    'een': BoysHandyJastrowKernel},
                                                jastrow_kernel_kwargs={
                                                    'ee': {'w': 1.},
                                                    'en': {'w': 1.},
                                                    'een': {}})

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight
        self.nbatch = 11
        self.pos = torch.Tensor(np.random.rand(
            self.nbatch,  self.wf.nelec*3))
        self.pos.requires_grad = True

    def test_forward(self):
        wfvals = self.wf(self.pos)

    def test_antisymmetry(self):
        """Test that the wf values are antisymmetric
        wrt exchange of 2 electrons of same spin."""
        wfvals_ref = self.wf(self.pos)

        if self.wf.nelec < 4:
            print(
                'Warning : antisymmetry cannot be tested with \
                    only %d electrons' % self.wf.nelec)
            return

        # test spin up
        pos_xup = self.pos.clone()
        perm_up = list(range(self.wf.nelec))
        perm_up[0] = 1
        perm_up[1] = 0
        pos_xup = pos_xup.reshape(self.nbatch, self.wf.nelec, 3)
        pos_xup = pos_xup[:, perm_up, :].reshape(
            self.nbatch, self.wf.nelec*3)

        wfvals_xup = self.wf(pos_xup)
        assert(torch.allclose(wfvals_ref, -1.*wfvals_xup))

    def test_jacobian_mo(self):
        """Jacobian of the BF MOs."""

        mo = self.wf.pos2mo(self.pos)
        dmo = self.wf.pos2mo(self.pos, derivative=1)

        dmo_grad = grad(
            mo, self.pos, grad_outputs=torch.ones_like(mo))[0]
        assert(torch.allclose(dmo.sum(), dmo_grad.sum()))

        psum_mo = dmo.sum(-1).sum(-1)
        psum_mo_grad = dmo_grad.view(
            self.nbatch, self.wf.nelec, 3).sum(-1)
        psum_mo_grad = psum_mo_grad.T
        assert(torch.allclose(psum_mo, psum_mo_grad))

    def test_grad_mo(self):
        """Gradients of the BF MOs."""

        mo = self.wf.pos2mo(self.pos)

        dao = self.wf.ao(self.pos, derivative=1, sum_grad=False)
        dmo = self.wf.ao2mo(dao)

        dmo_grad = grad(
            mo, self.pos,
            grad_outputs=torch.ones_like(mo))[0]
        assert(torch.allclose(dmo.sum(), dmo_grad.sum()))

        dmo = dmo.sum(-1).sum(-1)
        dmo_grad = dmo_grad.T

        assert(torch.allclose(dmo, dmo_grad))

    def test_hess_mo(self):
        """Hessian of the MOs."""
        val = self.wf.pos2mo(self.pos)

        d2val_grad = hess(val, self.pos)
        d2ao = self.wf.ao(self.pos, derivative=2, sum_hess=False)
        d2val = self.wf.ao2mo(d2ao)

        assert(torch.allclose(d2val.sum(), d2val_grad.sum()))

        d2val = d2val.reshape(4, 3, 11, 4, 3).sum(1).sum(-1).sum(-1)
        d2val_grad = d2val_grad.view(
            self.nbatch, self.wf.nelec, 3).sum(-1)
        d2val_grad = d2val_grad.T
        assert(torch.allclose(d2val, d2val_grad))

    def test_grad_wf(self):
        pass

        # grad_auto = self.wf.gradients_autograd(self.pos)
        # grad_jac = self.wf.gradients_jacobi(self.pos)

        # assert torch.allclose(
        #     grad_auto.data, grad_jac.data, rtol=1E-4, atol=1E-4)

    def test_local_energy(self):

        self.wf.kinetic_energy = self.wf.kinetic_energy_autograd
        eloc_auto = self.wf.local_energy(self.pos)

        self.wf.kinetic_energy = self.wf.kinetic_energy_jacobi
        eloc_jac = self.wf.local_energy(self.pos)

        assert torch.allclose(
            eloc_auto.data, eloc_jac.data, rtol=1E-4, atol=1E-4)

    def test_kinetic_energy(self):

        eauto = self.wf.kinetic_energy_autograd(self.pos)
        ejac = self.wf.kinetic_energy_jacobi(self.pos)

        print(ejac)
        print(eauto)

        assert torch.allclose(
            eauto.data, ejac.data, rtol=1E-4, atol=1E-4)


if __name__ == "__main__":
    # unittest.main()
    t = TestSlaterCombinedJastrowBackflow()
    t.setUp()
    t.test_hess_mo()
    # t.test_antisymmetry()
    # t.test_kinetic_energy()
