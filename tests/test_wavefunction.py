from qmctorch.wavefunction import Molecule, Orbital
from qmctorch.utils import set_torch_double_precision
import numpy as np
import torch
import unittest


class TestWaveFunction(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(101)
        set_torch_double_precision()

        # molecule
        mol = Molecule(
            atom='H 0 0 -0.69; H 0 0 0.69',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g')

        self.wf = Orbital(mol, kinetic='auto')

    def test_forward(self):
        pos = torch.rand(10, 6)
        wfvals = self.wf(pos)
        ref = torch.tensor([[0.0884],
                            [0.0799],
                            [0.1220],
                            [0.1299],
                            [0.1317],
                            [0.0912],
                            [0.1853],
                            [0.1409],
                            [0.1112],
                            [0.1397]])
        assert torch.allclose(wfvals.data, ref, rtol=1E-4, atol=1E-4)

    def test_local_energy(self):
        pos = torch.rand(10, 6)
        pos.requires_grad = True
        eloc_auto = self.wf.local_energy(pos)
        eloc_jac = self.wf.local_energy_jacobi(pos)

        ref = torch.tensor([[0.4716],
                            [-1.2498],
                            [-0.7811],
                            [-1.1405],
                            [-2.8688],
                            [-0.7188],
                            [-0.7862],
                            [-0.8492],
                            [-1.2014],
                            [-0.5513]])

        assert torch.allclose(
            eloc_auto.data, ref, rtol=1E-4, atol=1E-4)
        assert torch.allclose(
            eloc_auto.data, eloc_jac.data, rtol=1E-4, atol=1E-4)

    def test_kinetic_energy(self):
        pos = torch.rand(10, 6)
        pos.requires_grad = True
        eauto = self.wf.kinetic_energy_autograd(pos)
        ejac = self.wf.kinetic_energy_jacobi(pos)
        assert torch.allclose(
            eauto.data, ejac.data, rtol=1E-4, atol=1E-4)


if __name__ == "__main__":
    # unittest.main()
    t = TestWaveFunction()
    t.setUp()
    t.test_forward()
    t.test_local_energy()
    t.test_kinetic_energy()
