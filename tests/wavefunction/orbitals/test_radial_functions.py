import unittest
from qmctorch.wavefunction.orbitals.radial_functions import (radial_gaussian_pure,
                                                             radial_gaussian,
                                                             radial_slater,
                                                             radial_slater_pure)
import torch


class TestRadialFunctions(unittest.TestCase):

    def setUp(self):
        self.radfn = [radial_gaussian,
                      radial_gaussian_pure,
                      radial_slater,
                      radial_slater_pure]

        self.nbatch = 5
        self.nelec = 4
        self.nbas = 6

        self.bas_n = torch.Tensor([0, 1, 1, 1, 2, 2])
        self.bas_exp = torch.rand(self.nbas)

        self.xyz = torch.rand(self.nbatch, self.nelec, 3)

    def process_position(self):
        bas_coords = torch.zeros(self.nbas, 3)
        ndim = 3
        xyz = (self.xyz.view(-1, self.nelec, 1, ndim) -
               bas_coords[None, ...])

        r = torch.sqrt((xyz*xyz).sum(3))

        return xyz, r

    def test_val(self):
        xyz, r = self.process_position()
        for fn in self.radfn:
            fn(r, self.bas_n, self.bas_exp)

    def test_grad(self):
        xyz, r = self.process_position()
        for fn in self.radfn:
            fn(r, self.bas_n, self.bas_exp, xyz=xyz,
               derivative=1, sum_grad=False)

    def test_jac(self):
        xyz, r = self.process_position()
        for fn in self.radfn:
            fn(r, self.bas_n, self.bas_exp, xyz=xyz,
               derivative=1, sum_grad=True)

    def test_lap(self):
        xyz, r = self.process_position()
        for fn in self.radfn:
            fn(r, self.bas_n, self.bas_exp, xyz=xyz,
               derivative=2, sum_grad=True)

    def test_lap_sum(self):
        xyz, r = self.process_position()
        for fn in self.radfn:

            a = fn(r, self.bas_n, self.bas_exp, xyz=xyz,
                   derivative=2, sum_hess=False)
            b = fn(r, self.bas_n, self.bas_exp, xyz=xyz,
                   derivative=2, sum_hess=True)

            assert(torch.allclose(a.sum(-1), b, atol=1E-6))


if __name__ == "__main__":
    # unittest.TestCase()
    t = TestRadialFunctions()
    t.setUp()
    t.test_lap_sum()
