from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrowBackFlow
from qmctorch.utils import set_torch_double_precision

from torch.autograd import grad, gradcheck, Variable

import numpy as np
import torch
import unittest
import itertools
import os
import operator
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


class TestSlaterJastrowBackFlow(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        mol = Molecule(
            atom='H 0 0 0; H 0 0 3.015',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g',
            redo_scf=True)

        self.wf = SlaterJastrowBackFlow(mol,
                                        kinetic='auto',
                                        use_jastrow=False,
                                        include_all_mo=True,
                                        configs='ground_state')

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight

        self.wf.ao.backflow_weights.data += 0.1 * torch.rand(
            mol.nelec, mol.nelec)

        self.nbatch = 5
        self.pos = torch.Tensor(np.random.rand(
            self.nbatch,  self.wf.nelec*3))
        self.pos.requires_grad = True

    def test_forward(self):

        wfvals = self.wf(self.pos)
        # ref = torch.as_tensor([[0.0977],
        #                     [0.0618],
        #                     [0.0587],
        #                     [0.0861],
        #                     [0.0466],
        #                     [0.0406],
        #                     [0.0444],
        #                     [0.0728],
        #                     [0.0809],
        #                     [0.1868]])
        # assert torch.allclose(wfvals.data, ref, rtol=1E-4, atol=1E-4)

    def test_jacobian_mo(self):
        """Jacobian of the BF MOs."""

        mo = self.wf.pos2mo(self.pos)
        dmo = self.wf.pos2mo(self.pos, derivative=1)

        dmo_grad = grad(
            mo, self.pos, grad_outputs=torch.ones_like(mo))[0]
        assert(torch.allclose(dmo.sum(), dmo_grad.sum()))

        psum_mo = dmo.sum(-1).sum(0)
        psum_mo_grad = dmo_grad.view(
            self.nbatch, self.wf.nelec, 3).sum(-1)
        assert(torch.allclose(psum_mo, psum_mo_grad))

    def test_grad_mo(self):
        """Gradients of the BF MOs."""

        mo = self.wf.pos2mo(self.pos)
        dmo = self.wf.pos2mo(self.pos, derivative=1, jacobian=False)

        dmo_grad = grad(
            mo, self.pos,
            grad_outputs=torch.ones_like(mo))[0]
        assert(torch.allclose(dmo.sum(), dmo_grad.sum()))

        dmo = dmo.permute(1, 2, 3, 0)
        shape = (self.nbatch, self.wf.nelec,
                 self.wf.nmo_opt, self.wf.nelec, 3)
        dmo = dmo.reshape(*shape)
        dmo = dmo.sum(2).sum(1)

        dmo_grad = dmo_grad.reshape(self.nbatch, self.wf.nelec, 3)

        assert(torch.allclose(dmo, dmo_grad))

    def test_hess_mo(self):
        """Hessian of the MOs."""
        val = self.wf.pos2mo(self.pos)

        d2val_grad = hess(val, self.pos)
        d2val = self.wf.pos2mo(self.pos, derivative=2)
        assert(torch.allclose(d2val.sum(), d2val_grad.sum()))

        d2val = d2val.sum(0).sum(-1)
        d2val_grad = d2val_grad.view(
            self.nbatch, self.wf.nelec, 3).sum(-1)
        assert(torch.allclose(d2val, d2val_grad))

    def test_local_energy(self):

        self.wf.kinetic_energy = self.wf.kinetic_energy_autograd
        eloc_auto = self.wf.local_energy(self.pos)

        self.wf.kinetic_energy = self.wf.kinetic_energy_autograd
        eloc_jac = self.wf.local_energy(self.pos)

        assert torch.allclose(
            eloc_auto.data, eloc_jac.data, rtol=1E-4, atol=1E-4)

    def test_kinetic_energy(self):

        eauto = self.wf.kinetic_energy_autograd(self.pos)
        ejac = self.wf.kinetic_energy_jacobi(self.pos)
        print(eauto.sum())
        print(ejac.sum())
        assert torch.allclose(
            eauto.data, ejac.data, rtol=1E-4, atol=1E-4)

    def test_gradients_wf(self):

        grads = self.wf.gradients_jacobi(self.pos)
        grad_auto = self.wf.gradients_autograd(self.pos)

        assert torch.allclose(grads, grad_auto)

    def test_gradients_pdf(self):

        grads_pdf = self.wf.gradients_jacobi(self.pos, pdf=True)
        grads_auto = self.wf.gradients_autograd(self.pos, pdf=True)

        assert torch.allclose(grads_pdf, grads_auto)


if __name__ == "__main__":
    # unittest.main()
    t = TestSlaterJastrowBackFlow()
    t.setUp()
    # t.test_forward()
    # t.test_grad_mo()
    t.test_jacobian_mo()
    t.test_hess_mo()
    t.test_kinetic_energy()

    # wf = t.wf
    # x = t.pos
    # ao, dao, d2ao = wf.ao(x, derivative=[0, 1, 2],
    #                       jacobian=False)

    # # get the mo values
    # mo = wf.ao2mo(ao)
    # dmo = wf.ao2mo(dao)
    # d2mo = wf.ao2mo(d2ao)

    # # compute the value of the slater det
    # slater_dets = wf.pool(mo)
    # sum_slater_dets = wf.fc(slater_dets)

    # # compute ( tr(A_u^-1\Delta A_u) + tr(A_d^-1\Delta A_d) )
    # hess = wf.pool.operator(mo, d2mo)

    # # compute (tr(A_u^-1\nabla A_u) * tr(A_d^-1\nabla A_d))
    # grad = wf.pool.operator(mo, dmo, op=None)
    # grad2 = wf.pool.operator(mo, dmo, op_squared=True)

    # # assemble the total second derivative term
    # hess = (hess
    #         + operator.add(*[(g**2).sum(0) for g in grad])
    #         + 2 * operator.mul(*grad).sum(0)
    #         - grad2.sum(0))

    # hess = wf.fc(hess * slater_dets) / sum_slater_dets

    # # compute the Jastrow terms
    # jast, djast, d2jast = wf.jastrow(x, derivative=[0, 1, 2], jacobian=False)

    # djast = djast / jast.unsqueeze(-1)
    # djast = djast.permute(0, 2, 1).reshape(5, 12)

    # d2jast = d2jast / jast

    # grad_val = wf.fc(operator.add(*grad) *
    #                  slater_dets) / sum_slater_dets
    # grad_val = grad_val.squeeze().permute(1, 0)

    # d2jast.sum(-1) + 2*(grad_val * djast).sum(-1) + hess.squeeze().sum(0)
