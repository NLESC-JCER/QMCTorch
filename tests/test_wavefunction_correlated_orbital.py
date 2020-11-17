from qmctorch.scf import Molecule
from qmctorch.wavefunction import CorrelatedOrbital
from qmctorch.wavefunction import Orbital
from qmctorch.utils import set_torch_double_precision

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
            atom='H 0 0 0; H 0 0 1.',
            unit='bohr',
            calculator='pyscf',
            basis='dz',
            redo_scf=True)

        self.wf = CorrelatedOrbital(
            mol, kinetic='auto', configs='single_double(2,2)')

        # self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        # self.wf.fc.weight.data = self.random_fc_weight

        self.nbatch = 10
        self.pos = torch.tensor(np.random.rand(
            self.nbatch, self.wf.nelec*3))
        self.pos.requires_grad = True

    def test_forward(self):
        """Value of the wave function."""
        wfvals = self.wf(self.pos)

        ref = torch.tensor([[0.2339], [0.2092], [0.3335], [0.2806], [0.1317],
                            [0.0996], [0.1210], [0.1406], [0.2626], [0.4675]])

        # assert torch.allclose(wfvals.data, ref, rtol=1E-4, atol=1E-4)

    def test_grad_mo(self):
        """Gradients of the uncorrelated MOs."""
        mo = self.wf.pos2mo(self.pos)
        dmo = self.wf.pos2mo(self.pos, derivative=1)
        dmo_grad = grad(
            mo, self.pos, grad_outputs=torch.ones_like(mo))[0]

        gradcheck(self.wf.pos2mo, self.pos)

        assert(torch.allclose(dmo.sum(-1),
                              dmo_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1)))
        assert(torch.allclose(dmo.sum(), dmo_grad.sum()))

    def test_hess_mo(self):
        """Hessian of the uncorrelated MOs."""
        mo = self.wf.pos2mo(self.pos)
        d2mo = self.wf.pos2mo(self.pos, derivative=2)

        d2mo_grad = hess(mo, self.pos)

        assert(torch.allclose(d2mo.sum(), d2mo_grad.sum()))

    def test_grad_jast(self):
        """Gradients of the jastrow values."""

        jast = self.wf.jastrow(self.pos)
        djast = self.wf.jastrow(self.pos, derivative=1)

        djast_grad = grad(jast, self.pos,
                          grad_outputs=torch.ones_like(jast))[0]

        gradcheck(self.wf.jastrow, self.pos)

        assert(torch.allclose(djast_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1),
                              djast.sum(-1)))
        assert(torch.allclose(djast.sum(-1).sum(-1),
                              djast_grad.sum(-1)))
        assert(torch.allclose(djast.sum(),
                              djast_grad.sum()))

    def test_grad_cmo(self):
        """Gradients of the correlated MOs."""

        cmo = self.wf.pos2cmo(self.pos)
        dcmo = self.wf.pos2cmo(self.pos, derivative=1)

        dcmo_grad = grad(
            cmo,
            self.pos,
            grad_outputs=torch.ones_like(cmo))[0]

        gradcheck(self.wf.pos2cmo, self.pos)

        assert(torch.allclose(dcmo.sum(), dcmo_grad.sum()))
        assert(torch.allclose(dcmo.sum(-1),
                              dcmo_grad.view(10, 2, 3).sum(-1)))

    def test_hess_cmo(self):
        """Hessian of the correlated MOs."""
        val = self.wf.pos2cmo(self.pos)
        d2val_grad = hess(val, self.pos)
        d2val = self.wf.pos2cmo(self.pos, derivative=2)

        assert(torch.allclose(d2val.sum(), d2val_grad.sum()))

        assert(torch.allclose(d2val.sum(-1).sum(-1),
                              d2val_grad.view(10, 2, 3).sum(-1).sum(-1)))

        # assert(torch.allclose(d2val.sum(-1),
        #                       d2val_grad.view(10, 2, 3).sum(-1)))
    # def test_local_energy(self):

    #     self.wf.kinetic_energy = self.wf.kinetic_energy_autograd
    #     eloc_auto = self.wf.local_energy(self.pos)

    #     self.wf.kinetic_energy = self.wf.kinetic_energy_autograd
    #     eloc_jac = self.wf.local_energy(self.pos)

    #     ref = torch.tensor([[-1.6567],
    #                         [-0.8790],
    #                         [-2.8136],
    #                         [-0.3644],
    #                         [-0.4477],
    #                         [-0.2709],
    #                         [-0.6964],
    #                         [-0.3993],
    #                         [-0.4777],
    #                         [-0.0579]])

    #     assert torch.allclose(
    #         eloc_auto.data, ref, rtol=1E-4, atol=1E-4)

    #     assert torch.allclose(
    #         eloc_auto.data, eloc_jac.data, rtol=1E-4, atol=1E-4)

    def test_kinetic_energy(self):

        eauto = self.wf.kinetic_energy_autograd(self.pos)
        ejac = self.wf.kinetic_energy_jacobi(self.pos)

        # ref = torch.tensor([[0.6099],
        #                     [0.6438],
        #                     [0.6313],
        #                     [2.0512],
        #                     [0.0838],
        #                     [0.2699],
        #                     [0.5190],
        #                     [0.3381],
        #                     [1.8489],
        #                     [5.2226]])

        # assert torch.allclose(
        #     ejac.data, ref, rtol=1E-4, atol=1E-4)
        print(eauto)
        print(ejac)

        assert torch.allclose(
            eauto.data, ejac.data, rtol=1E-4, atol=1E-4)

    # def test_gradients_wf(self):

    #     grads = self.wf.gradients_jacobi(self.pos)
    #     grad_auto = self.wf.gradients_autograd(self.pos)

    #     assert torch.allclose(grads, grad_auto)

    # def test_gradients_pdf(self):

    #     grads_pdf = self.wf.gradients_jacobi(self.pos, pdf=True)
    #     grads_auto = self.wf.gradients_autograd(self.pos, pdf=True)

    #     assert torch.allclose(grads_pdf, grads_auto)


if __name__ == "__main__":

    set_torch_double_precision()

    # unittest.main()
    t = TestCorrelatedOrbitalWF()
    t.setUp()
    t.test_forward()
    t.test_grad_mo()
    t.test_hess_mo()
    t.test_grad_jast()
    # t.test_grad_cmo()
    t.test_hess_cmo()
    t.test_kinetic_energy()
    # t.test_local_energy()
