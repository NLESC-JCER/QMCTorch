import unittest

from qmctorch.sampler import GeneralizedMetropolis
from .test_sampler_base import TestSamplerBase


class TestGeneralizeMetropolis(TestSamplerBase):

    def test_gmh(self):
        """Test generalized MH."""
        sampler = GeneralizedMetropolis(
            nwalkers=10, nstep=20, step_size=0.2,
            nelec=self.wf.nelec, ndim=self.wf.ndim,
            init=self.mol.domain('normal'))

        pos = sampler(self.wf.pdf)


if __name__ == "__main__":
    unittest.main()
