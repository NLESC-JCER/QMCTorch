import unittest


from qmctorch.sampler import Metropolis
from .test_sampler_base import TestSamplerBase


class TestMetropolis(TestSamplerBase):

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


if __name__ == "__main__":
    unittest.main()
