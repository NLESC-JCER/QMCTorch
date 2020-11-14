from qmctorch.scf import Molecule
from qmctorch.wavefunction import Orbital
from qmctorch.utils import set_torch_double_precision
import numpy as np
import torch
import unittest
import itertools
import os


class TestOrbitalWF(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        mol = Molecule(
            atom='H 0 0 0; H 0 0 1.',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g',
            redo_scf=True)

        self.wf = Orbital(mol, kinetic='auto', configs='cas(2,2)')

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight

        self.pos = torch.tensor(np.random.rand(10, 6))
        self.pos.requires_grad = True

    def test_forward(self):

        wfvals = self.wf(self.pos)
        ref = torch.tensor([[0.0522],
                            [0.0826],
                            [0.0774],
                            [0.1321],
                            [0.0459],
                            [0.0421],
                            [0.0551],
                            [0.0764],
                            [0.1164],
                            [0.2506]])
        assert torch.allclose(wfvals.data, ref, rtol=1E-4, atol=1E-4)

    def test_local_energy(self):

        self.wf.kinetic_energy = self.wf.kinetic_energy_autograd
        eloc_auto = self.wf.local_energy(self.pos)

        self.wf.kinetic_energy = self.wf.kinetic_energy_autograd
        eloc_jac = self.wf.local_energy(self.pos)

        ref = torch.tensor([[-1.6567],
                            [-0.8790],
                            [-2.8136],
                            [-0.3644],
                            [-0.4477],
                            [-0.2709],
                            [-0.6964],
                            [-0.3993],
                            [-0.4777],
                            [-0.0579]])

        assert torch.allclose(
            eloc_auto.data, ref, rtol=1E-4, atol=1E-4)

        assert torch.allclose(
            eloc_auto.data, eloc_jac.data, rtol=1E-4, atol=1E-4)

    def test_kinetic_energy(self):

        eauto = self.wf.kinetic_energy_autograd(self.pos)
        ejac = self.wf.kinetic_energy_jacobi(self.pos)

        ref = torch.tensor([[0.6099],
                            [0.6438],
                            [0.6313],
                            [2.0512],
                            [0.0838],
                            [0.2699],
                            [0.5190],
                            [0.3381],
                            [1.8489],
                            [5.2226]])

        assert torch.allclose(
            ejac.data, ref, rtol=1E-4, atol=1E-4)

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
    # t = TestWaveFunction()
    # t.setUp()
    # t.test_forward()
    # t.test_local_energy()
    # t.test_kinetic_energy()
