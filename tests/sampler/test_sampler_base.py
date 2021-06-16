import unittest

import numpy as np
import torch

from qmctorch.utils import set_torch_double_precision
from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow


class TestSamplerBase(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        self.mol = Molecule(
            atom='H 0 0 -0.69; H 0 0 0.69',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g')

        # orbital
        self.wf = SlaterJastrow(self.mol)
