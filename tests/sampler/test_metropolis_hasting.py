import unittest
from qmctorch.sampler import MetropolisHasting
from qmctorch.sampler.metropolis_hasting import (
    ConstantVarianceKernel,
    CenterVarianceKernel,
)
from .test_sampler_base import TestSamplerBase


class TestMetropolisHasting(TestSamplerBase):
    def test_ConstantKernel(self):
        """Test Metropolis sampling."""

        sampler = MetropolisHasting(
            nwalkers=10,
            nstep=20,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain("normal"),
            kernel=ConstantVarianceKernel(),
        )

        _ = sampler(self.wf.pdf)

    def test_CenterVarianceKernel(self):
        """Test Metropolis sampling."""

        sampler = MetropolisHasting(
            nwalkers=10,
            nstep=20,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain("normal"),
            kernel=CenterVarianceKernel(),
        )

        _ = sampler(self.wf.pdf)


if __name__ == "__main__":
    unittest.main()
