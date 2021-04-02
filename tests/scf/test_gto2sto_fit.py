import unittest

import numpy as np
import torch

from qmctorch.utils import set_torch_double_precision
from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow


class TestGTO2STOFit(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        mol = Molecule(
            atom='C 0 0 0',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g',
            redo_scf=True)

        self.wf = SlaterJastrow(mol, kinetic='auto',
                                configs='ground_state').gto2sto()

        self.pos = -0.25 + 0.5 * \
            torch.as_tensor(np.random.rand(10, 18))
        self.pos.requires_grad = True

    def test_forward(self):

        wfvals = self.wf(self.pos)
        ref = torch.as_tensor([[-8.4430e-06],
                               [1.5092e-02],
                               [3.3809e-03],
                               [9.7981e-03],
                               [-6.8513e-02],
                               [-4.6836e-03],
                               [-3.2847e-04],
                               [2.3636e-02],
                               [5.5934e-04],
                               [1.3205e-02]])
        assert torch.allclose(wfvals.data, ref, rtol=1E-4, atol=1E-4)


if __name__ == "__main__":
    unittest.main()
