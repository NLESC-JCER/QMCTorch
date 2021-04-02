import unittest

import numpy as np
import torch

from qmctorch.sampler import Hamiltonian
from qmctorch.utils import set_torch_double_precision
from qmctorch.scf import Molecule
from qmctorch.wavefunction import Orbital
from test_sampler_base import TestSamplerBase


class TestHamiltonian(TestSamplerBase):

    def test_hmc(self):
        """Test HMC sampler."""
        sampler = Hamiltonian(
            nwalkers=10,
            nstep=20,
            step_size=0.1,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain('normal'))

        pos = sampler(self.wf.pdf)


if __name__ == "__main__":
    unittest.main()
