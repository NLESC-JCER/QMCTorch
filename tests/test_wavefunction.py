from qmctorch.wavefunction import Molecule, Orbital
from qmctorch.utils import set_torch_double_precision
import numpy as np
import torch
import unittest
import itertools


class TestWaveFunction(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        mol = Molecule(
            atom='H 0 0 -0.69; H 0 0 0.69',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g')

        self.wf = Orbital(mol, kinetic='auto', configs='cas(2,2)')
        self.pos = torch.tensor(np.random.rand(10, 6))
        self.pos.requires_grad = True

    def test_forward(self):

        wfvals = self.wf(self.pos)
        ref = torch.tensor([[0.1514],
                            [0.1427],
                            [0.2215],
                            [0.1842],
                            [0.0934],
                            [0.0697],
                            [0.0791],
                            [0.0963],
                            [0.1722],
                            [0.2988]])
        assert torch.allclose(wfvals.data, ref, rtol=1E-4, atol=1E-4)

    def test_local_energy(self):

        eloc_auto = self.wf.local_energy(self.pos)
        eloc_jac = self.wf.local_energy_jacobi(self.pos)
        ref = torch.tensor([[-1.3205],
                            [-1.1108],
                            [-1.2487],
                            [-1.0767],
                            [-0.4658],
                            [-0.2524],
                            [-0.6561],
                            [-0.5626],
                            [-1.1561],
                            [-1.9689]])

        assert torch.allclose(
            eloc_auto.data, ref, rtol=1E-4, atol=1E-4)

        assert torch.allclose(
            eloc_auto.data, eloc_jac.data, rtol=1E-4, atol=1E-4)

    def test_kinetic_energy(self):

        eauto = self.wf.kinetic_energy_autograd(self.pos)
        ejac = self.wf.kinetic_energy_jacobi(self.pos)
        ref = torch.tensor([[0.9461],
                            [0.4120],
                            [2.1962],
                            [1.3390],
                            [0.0657],
                            [0.2884],
                            [0.5593],
                            [0.1747],
                            [1.1705],
                            [3.3116]])

        assert torch.allclose(
            ejac.data, ref, rtol=1E-4, atol=1E-4)

        assert torch.allclose(
            eauto.data, ejac.data, rtol=1E-4, atol=1E-4)

        efd = self.wf.kinetic_energy_finite_difference(self.pos)


if __name__ == "__main__":
    unittest.main()
