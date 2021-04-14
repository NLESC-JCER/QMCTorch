from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrowOrbital
from qmctorch.wavefunction import SlaterJastrow
from qmctorch.utils import set_torch_double_precision, btrace

from qmctorch.wavefunction.jastrows.elec_elec.fully_connected_jastrow import FullyConnectedJastrow

from torch.autograd import grad, gradcheck, Variable

import numpy as np
import torch
import unittest
import itertools
import os


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


class TestCorrelatedOrbitalWF(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        mol = Molecule(
            atom='Li 0 0 0; H 0 0 1.',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g',
            redo_scf=True)

        self.wf = SlaterJastrowOrbital(mol,
                                       kinetic='auto',
                                       configs='ground_state',
                                       jastrow_type=FullyConnectedJastrow)

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight

        self.nbatch = 10
        self.pos = torch.as_tensor(np.random.rand(
            self.nbatch, self.wf.nelec*3))
        self.pos.requires_grad = True

    def test_forward(self):
        """Value of the wave function."""
        wfvals = self.wf(self.pos)

        ref = torch.as_tensor([[0.2339], [0.2092], [0.3335], [0.2806], [0.1317],
                               [0.0996], [0.1210], [0.1406], [0.2626], [0.4675]])

        # assert torch.allclose(wfvals.data, ref, rtol=1E-4, atol=1E-4)

    def test_jacobian_mo(self):
        """Jacobian of the uncorrelated MOs."""
        mo = self.wf.pos2mo(self.pos)
        dmo = self.wf.pos2mo(self.pos, derivative=1)
        dmo_grad = grad(
            mo, self.pos, grad_outputs=torch.ones_like(mo))[0]

        assert(torch.allclose(dmo.sum(-1),
                              dmo_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1)))

    def test_grad_mo(self):
        """Gradients of the uncorrelated MOs."""
        mo = self.wf.pos2mo(self.pos)
        dmo = self.wf.pos2mo(self.pos, derivative=1, jacobian=False)
        dmo_grad = grad(
            mo, self.pos, grad_outputs=torch.ones_like(mo))[0]

        assert(torch.allclose(dmo.sum(-2),
                              dmo_grad.view(self.nbatch, self.wf.nelec, 3)))

    def test_hess_mo(self):
        """Hessian of the uncorrelated MOs."""
        mo = self.wf.pos2mo(self.pos)
        d2mo = self.wf.pos2mo(self.pos, derivative=2)
        d2mo_grad = hess(mo, self.pos)

        assert(torch.allclose(d2mo.sum(-1),
                              d2mo_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1)))

    def test_jacobian_jast(self):
        """Jacobian of the jastrow values."""
        jast = self.wf.ordered_jastrow(self.pos)
        djast = self.wf.ordered_jastrow(self.pos, derivative=1)
        djast_grad = grad(jast, self.pos,
                          grad_outputs=torch.ones_like(jast))[0]

        assert(torch.allclose(djast_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1),
                              djast.sum(-1)))

    def test_grad_jast(self):
        """Gradients of the jastrow values."""
        jast = self.wf.ordered_jastrow(self.pos)
        djast = self.wf.ordered_jastrow(
            self.pos, derivative=1, jacobian=False)
        djast_grad = grad(jast, self.pos,
                          grad_outputs=torch.ones_like(jast))[0]

        assert(torch.allclose(djast_grad.view(self.nbatch, self.wf.nelec, 3),
                              djast.sum(-2)))

    def test_hess_jast(self):
        """Hessian of the jastrows."""
        jast = self.wf.ordered_jastrow(self.pos)
        d2jast = self.wf.ordered_jastrow(self.pos, derivative=2)

        d2jast_grad = hess(jast, self.pos)

        assert(torch.allclose(d2jast.sum(-1),
                              d2jast_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1)))

    def test_grad_cmo(self):
        """Gradients of the correlated MOs."""
        cmo = self.wf.pos2cmo(self.pos)
        dcmo = self.wf.get_gradient_operator(self.pos)

        dcmo = dcmo.permute(1, 2, 3, 0)
        shape = (self.nbatch, self.wf.nelec,
                 self.wf.nmo_opt, self.wf.nelec, 3)
        dcmo = dcmo.reshape(*shape)
        dcmo = dcmo.sum(2).sum(1)

        dcmo_grad = grad(cmo, self.pos,
                         grad_outputs=torch.ones_like(cmo))[0]
        dcmo_grad = dcmo_grad.reshape(self.nbatch, self.wf.nelec, 3)

        assert(torch.allclose(dcmo, dcmo_grad))

    def test_hess_cmo(self):
        """Hessian of the correlated MOs."""
        val = self.wf.pos2cmo(self.pos)
        d2val_grad = hess(val, self.pos)

        d2val = self.wf.get_hessian_operator(self.pos)
        d2val = d2val.permute(1, 2, 0, 3).sum(1)

        assert(torch.allclose(d2val.sum(-1),
                              d2val_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1)))

    def test_jacobian_wf(self):
        """Jacobian of det(CMO).
            \nabla det(CMOup) / det(CMOup) +  \nabla det(CMOup) / det(CMOup) """
        grad_jacobi = self.wf.gradients_jacobi(self.pos)
        grad_auto = self.wf.gradients_autograd(self.pos)
        assert(torch.allclose(grad_jacobi, grad_auto.sum(-1)))

    def test_grad_wf(self):
        """Compute the gradients of the wf  wrt to xyz coord of each elec."""
        grad_jacobi = self.wf.gradients_jacobi(
            self.pos, jacobian=False).squeeze()
        grad_auto = self.wf.gradients_autograd(self.pos)
        assert torch.allclose(grad_jacobi, grad_auto)

    def test_kinetic_energy(self):
        """Kinetic energty."""
        eauto = self.wf.kinetic_energy_autograd(self.pos)
        ejac = self.wf.kinetic_energy_jacobi(self.pos)

        assert torch.allclose(
            eauto.data, ejac.data, rtol=1E-4, atol=1E-4)

    def test_local_energy(self):
        """local energy."""
        self.wf.kinetic_energy = self.wf.kinetic_energy_autograd
        eloc_auto = self.wf.local_energy(self.pos)

        self.wf.kinetic_energy = self.wf.kinetic_energy_autograd
        eloc_jac = self.wf.local_energy(self.pos)

        assert torch.allclose(
            eloc_auto.data, eloc_jac.data, rtol=1E-4, atol=1E-4)


if __name__ == "__main__":

    set_torch_double_precision()

    t = TestCorrelatedOrbitalWF()
    t.setUp()
    t.test_forward()

    t.test_jacobian_mo()
    t.test_grad_mo()
    t.test_hess_mo()

    t.test_jacobian_jast()
    t.test_grad_jast()
    t.test_hess_jast()

    t.test_grad_cmo()
    t.test_hess_cmo()

    t.test_jacobian_wf()
    t.test_grad_wf()

    t.test_kinetic_energy()
    t.test_local_energy()