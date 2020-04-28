from qmctorch.wavefunction import Molecule
from qmctorch.wavefunction.orbital_configurations import OrbitalConfigurations
from qmctorch.utils import set_torch_double_precision
import numpy as np
import torch
import unittest
import itertools


class TestOrbitalConfiguration(unittest.TestCase):

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

        self.orb_conf = OrbitalConfigurations(mol)

    def test_confs(self):

        self.orb_conf.get_configs('ground_state')
        self.orb_conf.get_configs('single(2,2)')
        self.orb_conf.get_configs('single_double(2,2)')
        self.orb_conf.get_configs('cas(2,2)')


if __name__ == "__main__":
    unittest.main()
