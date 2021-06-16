from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow
from qmctorch.utils import set_torch_double_precision

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


class TestOrbitalWF(unittest.TestCase):

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

        self.wf = SlaterJastrow(mol,
                                kinetic='auto',
                                include_all_mo=True,
                                configs='cas(2,2)')

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight

        self.nbatch = 10
        self.pos = torch.Tensor(
            np.random.rand(self.nbatch, mol.nelec*3))
        self.pos.requires_grad = True

    def test_forward(self):

        wfvals = self.wf(self.pos)

        ref = torch.Tensor([[0.0522],
                            [0.0826],
                            [0.0774],
                            [0.1321],
                            [0.0459],
                            [0.0421],
                            [0.0551],
                            [0.0764],
                            [0.1164],
                            [0.2506]])
        # assert torch.allclose(wfvals.data, ref, rtol=1E-4, atol=1E-4)

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
        assert(torch.allclose(wfvals_ref, -1*wfvals_xup))

        # test spin down
        pos_xdn = self.pos.clone()
        perm_dn = list(range(self.wf.nelec))
        perm_dn[self.wf.mol.nup-1] = self.wf.mol.nup
        perm_dn[self.wf.mol.nup] = self.wf.mol.nup-1
        pos_xdn = pos_xdn.reshape(self.nbatch, self.wf.nelec, 3)
        pos_xdn = pos_xdn[:, perm_up, :].reshape(
            self.nbatch, self.wf.nelec*3)

        wfvals_xdn = self.wf(pos_xdn)
        assert(torch.allclose(wfvals_ref, -1*wfvals_xdn))

    def test_grad_mo(self):
        """Gradients of the MOs."""

        mo = self.wf.pos2mo(self.pos)
        dmo = self.wf.pos2mo(self.pos, derivative=1)

        dmo_grad = grad(
            mo,
            self.pos,
            grad_outputs=torch.ones_like(mo))[0]

        gradcheck(self.wf.pos2mo, self.pos)

        assert(torch.allclose(dmo.sum(), dmo_grad.sum()))
        assert(torch.allclose(dmo.sum(-1),
                              dmo_grad.view(10, self.wf.nelec, 3).sum(-1)))

    def test_hess_mo(self):
        """Hessian of the MOs."""
        val = self.wf.pos2mo(self.pos)

        d2val_grad = hess(val, self.pos)
        d2val = self.wf.pos2mo(self.pos, derivative=2)

        assert(torch.allclose(d2val.sum(), d2val_grad.sum()))

        assert(torch.allclose(d2val.sum(-1).sum(-1),
                              d2val_grad.view(10, self.wf.nelec, 3).sum(-1).sum(-1)))

        assert(torch.allclose(d2val.sum(-1),
                              d2val_grad.view(10, self.wf.nelec, 3).sum(-1)))

    def test_local_energy(self):

        self.wf.kinetic_energy = self.wf.kinetic_energy_autograd
        eloc_auto = self.wf.local_energy(self.pos)

        self.wf.kinetic_energy = self.wf.kinetic_energy_autograd
        eloc_jac = self.wf.local_energy(self.pos)

        ref = torch.Tensor([[-1.6567],
                            [-0.8790],
                            [-2.8136],
                            [-0.3644],
                            [-0.4477],
                            [-0.2709],
                            [-0.6964],
                            [-0.3993],
                            [-0.4777],
                            [-0.0579]])

        # assert torch.allclose(
        #     eloc_auto.data, ref, rtol=1E-4, atol=1E-4)

        assert torch.allclose(
            eloc_auto.data, eloc_jac.data, rtol=1E-4, atol=1E-4)

    def test_kinetic_energy(self):

        eauto = self.wf.kinetic_energy_autograd(self.pos)
        ejac = self.wf.kinetic_energy_jacobi(self.pos)

        ref = torch.Tensor([[0.6099],
                            [0.6438],
                            [0.6313],
                            [2.0512],
                            [0.0838],
                            [0.2699],
                            [0.5190],
                            [0.3381],
                            [1.8489],
                            [5.2226]])

        # assert torch.allclose(
        #     ejac.data, ref, rtol=1E-4, atol=1E-4)

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
    unittest.main()
