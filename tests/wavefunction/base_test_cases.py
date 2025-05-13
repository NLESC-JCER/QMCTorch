import unittest
from torch.autograd import grad, gradcheck
import torch
from qmctorch.utils.torch_utils import diagonal_hessian as hess


class BaseTestCases:
    class WaveFunctionBaseTest(unittest.TestCase):
        def setUp(self):
            """Init the base test"""

            def wf_placeholder(pos, **kwargs):
                """Callable for wf"""
                return None

            self.pos = None
            self.wf = wf_placeholder
            self.nbatch = None

        def test_forward(self):
            """Test that the forward pass works"""
            _ = self.wf(self.pos)

        def test_antisymmetry(self):
            """Test that the wf values are antisymmetric
            wrt exchange of 2 electrons of same spin."""
            wfvals_ref = self.wf(self.pos)

            if self.wf.nelec < 4:
                print(
                    "Warning : antisymmetry cannot be tested with \
                        only %d electrons"
                    % self.wf.nelec
                )
                return

            # test spin up
            pos_xup = self.pos.clone()
            perm_up = list(range(self.wf.nelec))
            perm_up[0] = 1
            perm_up[1] = 0
            pos_xup = pos_xup.reshape(self.nbatch, self.wf.nelec, 3)
            pos_xup = pos_xup[:, perm_up, :].reshape(self.nbatch, self.wf.nelec * 3)

            wfvals_xup = self.wf(pos_xup)
            assert torch.allclose(wfvals_ref, -1 * wfvals_xup)

            # test spin down
            pos_xdn = self.pos.clone()
            perm_dn = list(range(self.wf.nelec))
            perm_dn[self.wf.mol.nup - 1] = self.wf.mol.nup
            perm_dn[self.wf.mol.nup] = self.wf.mol.nup - 1
            pos_xdn = pos_xdn.reshape(self.nbatch, self.wf.nelec, 3)
            pos_xdn = pos_xdn[:, perm_up, :].reshape(self.nbatch, self.wf.nelec * 3)

            wfvals_xdn = self.wf(pos_xdn)
            assert torch.allclose(wfvals_ref, -1 * wfvals_xdn)

        def test_grad_mo(self):
            """Gradients of the MOs."""

            mo = self.wf.pos2mo(self.pos)
            dmo = self.wf.pos2mo(self.pos, derivative=1)

            dmo_grad = grad(mo, self.pos, grad_outputs=torch.ones_like(mo))[0]

            gradcheck(self.wf.pos2mo, self.pos)

            assert torch.allclose(dmo.sum(), dmo_grad.sum())
            assert torch.allclose(
                dmo.sum(-1), dmo_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1)
            )

        def test_hess_mo(self):
            """Hessian of the MOs."""
            val = self.wf.pos2mo(self.pos)

            d2val_grad, _ = hess(val, self.pos)
            d2val = self.wf.pos2mo(self.pos, derivative=2)

            assert torch.allclose(d2val.sum(), d2val_grad.sum())

            assert torch.allclose(
                d2val.sum(-1).sum(-1),
                d2val_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1).sum(-1),
            )

            assert torch.allclose(
                d2val.sum(-1), d2val_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1)
            )

        def test_local_energy(self):
            self.wf.kinetic_energy = self.wf.kinetic_energy_autograd
            eloc_auto = self.wf.local_energy(self.pos)

            self.wf.kinetic_energy = self.wf.kinetic_energy_jacobi
            eloc_jac = self.wf.local_energy(self.pos)

            assert torch.allclose(eloc_auto.data, eloc_jac.data, rtol=1e-4, atol=1e-4)

        def test_kinetic_energy(self):
            eauto = self.wf.kinetic_energy_autograd(self.pos)
            ejac = self.wf.kinetic_energy_jacobi(self.pos)

            assert torch.allclose(eauto.data, ejac.data, rtol=1e-4, atol=1e-4)

        def test_gradients_wf(self):
            grads = self.wf.gradients_jacobi(self.pos, sum_grad=False).squeeze()
            grad_auto = self.wf.gradients_autograd(self.pos)

            assert torch.allclose(grads.sum(), grad_auto.sum())

            grads = grads.reshape(self.nbatch, self.wf.nelec, 3)
            grad_auto = grad_auto.reshape(self.nbatch, self.wf.nelec, 3)
            assert torch.allclose(grads, grad_auto)

        def test_gradients_pdf(self):
            grads_pdf = self.wf.gradients_jacobi(self.pos, pdf=True)
            grads_auto = self.wf.gradients_autograd(self.pos, pdf=True)

            assert torch.allclose(grads_pdf.sum(), grads_auto.sum())

    class BackFlowWaveFunctionBaseTest(WaveFunctionBaseTest):
        def test_jacobian_mo(self):
            """Jacobian of the BF MOs."""

            mo = self.wf.pos2mo(self.pos)
            dmo = self.wf.pos2mo(self.pos, derivative=1)

            dmo_grad = grad(mo, self.pos, grad_outputs=torch.ones_like(mo))[0]
            assert torch.allclose(dmo.sum(), dmo_grad.sum())

            psum_mo = dmo.sum(-1).sum(-1)
            psum_mo_grad = dmo_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1)
            psum_mo_grad = psum_mo_grad.T
            assert torch.allclose(psum_mo, psum_mo_grad)

        def test_grad_mo(self):
            """Gradients of the BF MOs."""

            mo = self.wf.pos2mo(self.pos)

            dao = self.wf.ao(self.pos, derivative=1, sum_grad=False)
            dmo = self.wf.ao2mo(dao)

            dmo_grad = grad(mo, self.pos, grad_outputs=torch.ones_like(mo))[0]
            assert torch.allclose(dmo.sum(), dmo_grad.sum())

            dmo = dmo.sum(-1).sum(-1)
            dmo_grad = dmo_grad.T

            assert torch.allclose(dmo, dmo_grad)

        def test_hess_mo(self):
            """Hessian of the MOs."""
            val = self.wf.pos2mo(self.pos)

            d2val_grad, _ = hess(val, self.pos)
            d2ao = self.wf.ao(self.pos, derivative=2, sum_hess=False)
            d2val = self.wf.ao2mo(d2ao)

            assert torch.allclose(d2val.sum(), d2val_grad.sum())

            d2val = d2val.reshape(4, 3, 5, 4, 6).sum(1).sum(-1).sum(-1)
            d2val_grad = d2val_grad.view(self.nbatch, self.wf.nelec, 3).sum(-1)
            d2val_grad = d2val_grad.T
            assert torch.allclose(d2val, d2val_grad)

        def test_gradients_wf(self):
            pass

        def test_gradients_pdf(self):
            pass
