import unittest
from qmctorch.wavefunction.orbitals.radial_functions import (radial_gaussian_pure,
                                                             radial_gaussian,
                                                             radial_slater,
                                                             radial_slater_pure)
import torch
from torch.autograd import grad, Variable


def hess(out, pos):

    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos,
                 grad_outputs=z,
                 only_inputs=True,
                 create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[1]):

        tmp = grad(jacob[:, idim], pos,
                   grad_outputs=z,
                   only_inputs=True,
                   create_graph=True)[0]

        hess[:, idim] = tmp[:, idim]

    return hess


def hess_mixed_terms(out, pos):

    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos,
                 grad_outputs=z,
                 only_inputs=True,
                 create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)
    nelec = pos.shape[1]//3
    k = 0

    for ielec in range(nelec):

        ix = ielec*3
        tmp = grad(jacob[:, ix], pos,
                   grad_outputs=z,
                   only_inputs=True,
                   create_graph=True)[0]

        hess[:, k] = tmp[:, ix+1]
        k = k + 1
        hess[:, k] = tmp[:, ix+2]
        k = k + 1

        iy = ielec*3 + 1
        tmp = grad(jacob[:, iy], pos,
                   grad_outputs=z,
                   only_inputs=True,
                   create_graph=True)[0]

        hess[:, k] = tmp[:, iy+1]
        k = k + 1

    return hess


class TestRadialFunctions(unittest.TestCase):

    def setUp(self):
        self.radfn = [radial_gaussian,
                      radial_gaussian_pure,
                      radial_slater,
                      radial_slater_pure]

        self.nbatch = 10
        self.nelec = 4
        self.nbas = 6

        self.bas_n = torch.Tensor([0, 1, 1, 1, 2, 2])
        self.bas_exp = torch.rand(self.nbas)

        self.xyz = Variable(torch.rand(self.nbatch, self.nelec*3))
        self.xyz.requires_grad = True

    def process_position(self):
        """Return the distance between electron and centers."""

        bas_coords = torch.zeros(self.nbas, 3)
        xyz = (self.xyz.view(-1, self.nelec, 1, 3) -
               bas_coords[None, ...])
        r = torch.sqrt((xyz*xyz).sum(3))
        return xyz, r

    def test_val(self):
        """Simply executes the kernel."""
        xyz, r = self.process_position()
        for fn in self.radfn:
            fn(r, self.bas_n, self.bas_exp)

    def test_grad(self):
        """Compute the gradients of the radial function."""

        for fn in self.radfn:

            xyz, r = self.process_position()
            val = fn(r, self.bas_n, self.bas_exp)
            val_grad_auto = grad(
                val, self.xyz, torch.ones_like(val))[0]

            val_grad = fn(r, self.bas_n, self.bas_exp, xyz=xyz,
                          derivative=1, sum_grad=False)

            val_grad_sum = fn(r, self.bas_n, self.bas_exp, xyz=xyz,
                              derivative=1, sum_grad=True)

            assert(torch.allclose(
                val_grad.sum(), val_grad_auto.sum(), atol=1E-6))

            assert(torch.allclose(
                val_grad.sum(-1), val_grad_sum, atol=1E-6))

    def test_lap(self):
        """Computes the laplacian of the radial functions."""

        for fn in self.radfn:

            xyz, r = self.process_position()
            val = fn(r, self.bas_n, self.bas_exp)

            val_lap = fn(r, self.bas_n, self.bas_exp, xyz=xyz,
                         derivative=2, sum_hess=False)
            val_lap_sum = fn(r, self.bas_n, self.bas_exp, xyz=xyz,
                             derivative=2, sum_hess=True)

            val_lap_auto = hess(val, self.xyz)

            assert(torch.allclose(
                val_lap.sum(-1), val_lap_sum, atol=1E-6))

            assert(torch.allclose(
                val_lap.sum(), val_lap_auto.sum(), atol=1E-6))

    def test_mixed(self):
        """Test the mixed second derivatives."""
        for fn in self.radfn:
            xyz, r = self.process_position()
            val = fn(r, self.bas_n, self.bas_exp)
            val_lap = fn(r, self.bas_n, self.bas_exp, xyz=xyz,
                         derivative=3)

        val_lap_auto = hess_mixed_terms(val, self.xyz)
        assert(torch.allclose(
            val_lap.sum(), val_lap_auto.sum(), atol=1E-6))


if __name__ == "__main__":
    # unittest.TestCase()
    t = TestRadialFunctions()
    t.setUp()
    t.test_grad()
    t.test_lap()
    t.test_mixed()
