from qmctorch.wavefunction import Molecule, Orbital
from qmctorch.sampler import Metropolis, Hamiltonian, GeneralizedMetropolis
from qmctorch.sampler.walkers import Walkers
from qmctorch.utils import set_torch_double_precision
import numpy as np
import torch
import unittest
import itertools


class TestSampler(unittest.TestCase):

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
        self.wf = Orbital(self.mol)

    def test_walkers_init(self):
        """Test different initialization methods of the walkers."""
        w1 = Walkers(nwalkers=10,
                     nelec=self.mol.nelec, ndim=3,
                     init=self.mol.domain('center'))

        w2 = Walkers(nwalkers=10,
                     nelec=self.mol.nelec, ndim=3,
                     init=self.mol.domain('uniform'))

        w3 = Walkers(nwalkers=10,
                     nelec=self.mol.nelec, ndim=3,
                     init=self.mol.domain('normal'))

        w4 = Walkers(nwalkers=10,
                     nelec=self.mol.nelec, ndim=3,
                     init=self.mol.domain('atomic'))

    def test_metropolis(self):
        """Test Metropolis sampling."""

        sampler = Metropolis(
            nwalkers=10,
            nstep=20,
            step_size=0.5,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain('normal'))

        for m in ['one-elec', 'all-elec', 'all-elec-iter']:
            for p in ['normal', 'uniform']:

                sampler.configure_move({'type': m, 'proba': p})
                pos = sampler(self.wf.pdf)

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

    def test_gmh(self):
        """Test generalized MH."""
        sampler = GeneralizedMetropolis(
            nwalkers=10, nstep=20, step_size=0.2,
            nelec=self.wf.nelec, ndim=self.wf.ndim,
            init=self.mol.domain('normal'))

        pos = sampler(self.wf.pdf)


if __name__ == "__main__":
    unittest.main()
