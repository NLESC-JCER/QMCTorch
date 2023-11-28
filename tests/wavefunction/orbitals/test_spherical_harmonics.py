import unittest

import numpy as np
import torch

from qmctorch.wavefunction.orbitals.spherical_harmonics import Harmonics


class TestSphericalHarmonics(unittest.TestCase):

    def setUp(self):
        bas_l = torch.Tensor([0, 1, 1, 1, 2, 2, 2, 2, 2])
        bas_m = torch.Tensor([0, -1, 0, 1, -2, -1, 0, 1, 2])

        self.harmonics = Harmonics('sph', bas_l=bas_l, bas_m=bas_m)
        self.pos = torch.rand(5, 4, 9, 3)

    def test_value(self):
        self.harmonics(self.pos, derivative=0)

    def test_grad(self):
        self.harmonics(self.pos, derivative=1, sum_grad=False)

    def test_jac(self):
        self.harmonics(self.pos, derivative=1, sum_grad=True)

    def test_lap(self):
        self.harmonics(self.pos, derivative=2)


if __name__ == "__main__":
    unittest.main()
