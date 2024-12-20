import unittest

import numpy as np
import torch

from qmctorch.utils import set_torch_double_precision
from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow

from qmctorch.wavefunction.jastrows.elec_elec.jastrow_factor_electron_electron import (
    JastrowFactorElectronElectron,
)
from qmctorch.wavefunction.jastrows.elec_elec.kernels import PadeJastrowKernel


class TestSamplerBase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        self.mol = Molecule(
            atom="H 0 0 -0.69; H 0 0 0.69",
            unit="bohr",
            calculator="pyscf",
            basis="sto-3g",
        )

        jastrow = JastrowFactorElectronElectron(self.mol, PadeJastrowKernel)

        # orbital
        self.wf = SlaterJastrow(self.mol, jastrow=jastrow)
