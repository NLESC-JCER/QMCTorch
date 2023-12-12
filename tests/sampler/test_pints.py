import unittest


import pints
from qmctorch.sampler import PintsSampler
from .test_sampler_base import TestSamplerBase


class TestPints(TestSamplerBase):
    def test_Haario(self):
        """Test Metropolis sampling."""

        sampler = PintsSampler(
            nwalkers=10,
            nstep=20,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain("normal"),
            method=pints.HaarioBardenetACMC,
        )

        _ = sampler(self.wf.pdf)

    def test_Langevin(self):
        """Test Metropolis sampling."""

        sampler = PintsSampler(
            nwalkers=10,
            nstep=20,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain("normal"),
            method=pints.MALAMCMC,
            method_requires_grad=True,
        )

        _ = sampler(self.wf.pdf)


if __name__ == "__main__":
    unittest.main()
