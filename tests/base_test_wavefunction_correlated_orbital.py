from qmctorch.scf import Molecule
from qmctorch.wavefunction import CorrelatedOrbital
from qmctorch.wavefunction import Orbital
from qmctorch.utils import set_torch_double_precision, btrace

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


class BaseTestCorrelatedOrbitalWF(unittest.TestCase):

    def setUp(self):
        raise NotImplementedError('Please Implement a setUp function')

    def test_forward(self):
        raise NotImplementedError(
            'Please Implement a forward function')

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
        print('d2val', d2val.shape)

        d2val = d2val.permute(1, 2, 0, 3).sum(1)
        print('d2val', d2val.shape)
        assert(torch.allclose(d2val.sum(-1),
                              d2val_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1)))

    def test_grad_manual(self):

        nup = self.wf.mol.nup

        # compute the cmos
        cmo = self.wf.pos2cmo(self.pos)
        aup = cmo[:, :nup, :nup]
        iaup = torch.inverse(aup)
        sd = torch.det(aup)

        # get the hessian of that
        grad_auto = grad(
            sd, self.pos, grad_outputs=torch.ones_like(sd))[0]
        grad_auto = grad_auto.sum(-1) / sd

        # get th matrix of hess
        opgrad = self.wf.get_gradient_operator(
            self.pos)
        print(opgrad.shape)
        opgrad_up = opgrad[:, :, :nup, :nup]
        grad_jacobi = btrace(iaup @ opgrad_up).sum(0)

        print(grad_auto.sum())
        print(grad_jacobi.sum())
        assert(torch.allclose(grad_jacobi, grad_auto))

    def test_hess_manual(self):

        nup = self.wf.mol.nup

        # compute the cmos
        cmo = self.wf.pos2cmo(self.pos)
        aup = cmo[:, :nup, :nup]
        iaup = torch.inverse(aup)
        sd = torch.det(aup)

        # get the hessian of that
        hess_auto = hess(sd, self.pos).sum(-1) / sd

        # get th matrix of hess
        ophess = self.wf.get_hessian_operator(
            self.pos)
        print(ophess.shape)
        ophess_up = ophess[:, :, :nup, :nup]
        hess_jacobi = btrace(iaup @ ophess_up).sum(0)

        # # get the matrix of hess
        # ophess = self.wf.pos2cmo(self.pos, derivative=2)
        # print(ophess.shape)
        # ophess_up = ophess[:, :nup, :nup]
        # hess_jacobi = btrace(iaup @ ophess_up)

        print(hess_auto.sum())
        print(hess_jacobi.sum())
        assert(torch.allclose(hess_jacobi, hess_auto))

    def test_grad_slater_det(self):
        """gradients of the slater determinant."""
        cmo = self.wf.pos2cmo(self.pos)
        sd = self.wf.pool(cmo)[:, 0]

        bgrad = self.wf.get_gradient_operator(self.pos)
        grad_gs = (self.wf.pool.operator(
            cmo, bgrad, op=operator.add).sum(0))[:, 0]

        sd_grad = grad(
            sd, self.pos, grad_outputs=torch.ones_like(sd))[0]
        sd_grad = sd_grad.sum(-1) / sd

        assert(torch.allclose(grad_gs, sd_grad))

    def test_hess_slater_det(self):
        """hessian of the slater determinant."""
        cmo = self.wf.pos2cmo(self.pos)
        sd = self.wf.pool(cmo)[:, 0]

        bhess = self.wf.get_hessian_operator(self.pos)
        print(bhess.shape)
        hess_gs = (self.wf.pool.operator(cmo, bhess).sum(0))[:, 0]
        print(hess_gs.shape)

        # bhess = self.wf.pos2cmo(self.pos, derivative=2)
        # print(bhess.shape)
        # hess_gs = self.wf.pool.operator(cmo, bhess)[:, 0]
        # print(hess_gs.shape)

        bgrad = self.wf.get_gradient_operator(self.pos)
        print(bgrad.shape)
        grad_gs = self.wf.pool.operator(cmo, bgrad, op=operator.mul)
        print(grad_gs.shape)
        grad_gs = grad_gs.sum(0)
        print(grad_gs.shape)
        grad_gs = grad_gs[:, 0]

        check = hess(sd, self.pos)
        check = check.sum(-1) / sd

        hess_jac = hess_gs+2*grad_gs
        print(hess_jac.sum())
        print(check.sum())
        assert(torch.allclose(hess_jac, check))

    def test_hess_slater_det_manual(self):

        nup = self.wf.mol.nup

        # compute the cmos
        cmo = self.wf.pos2cmo(self.pos)

        # extract up./down matrices
        aup = cmo[:, :nup, :nup]
        adown = cmo[:, nup:, :nup]

        # compute inverse
        iaup = torch.inverse(aup)
        iadown = torch.inverse(adown)

        # get the slater det prod
        sd = torch.det(aup)*torch.det(adown)

        # get the hessian of that
        hess_auto = hess(sd, self.pos).sum(-1) / sd

        # get the matrix of the gradients
        opgrad = self.wf.get_gradient_operator(self.pos)

        # get up/down op grad
        opgrad_up = opgrad[:, :, :nup, :nup]
        opgrad_down = opgrad[:, :, nup:, :nup]

        # get up/dpwn grad vals
        grad_up = btrace(iaup @ opgrad_up)
        grad_down = btrace(iadown @ opgrad_down)

        # get th matrix of hess
        ophess = self.wf.get_hessian_operator(self.pos)

        # get up/down hess op
        ophess_up = ophess[:, :, :nup, :nup]
        ophess_down = ophess[:, :, nup:, :nup]

        # get up/dpwn grad vals
        hess_up = btrace(iaup @ ophess_up)
        hess_down = btrace(iadown @ ophess_down)

        hess_jacobi = (hess_up.sum(0) + hess_down.sum(0) +
                       2 * (grad_up*grad_down).sum(0))

        print(hess_auto.sum())
        print(hess_jacobi.sum())
        assert(torch.allclose(hess_jacobi, hess_auto))

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
