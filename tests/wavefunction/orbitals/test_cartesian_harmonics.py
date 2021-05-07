import unittest

import numpy as np
import torch
from torch.autograd import grad, Variable

from qmctorch.wavefunction.orbitals.spherical_harmonics import Harmonics


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


class TestCartesianHarmonics(unittest.TestCase):

    def setUp(self):
        bas_kx = torch.as_tensor([0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1])
        bas_ky = torch.as_tensor([0, 0, 1, 0, 0, 2, 0, 1, 1, 0, 1])
        bas_kz = torch.as_tensor([0, 0, 0, 1, 0, 0, 2, 0, 1, 1, 1])
        self.nbas = len(bas_kx)

        self.harmonics = Harmonics(
            'cart', bas_kx=bas_kx, bas_ky=bas_ky, bas_kz=bas_kz)

        self.nbatch = 10
        self.nelec = 4

        self.pos = Variable(torch.rand(self.nbatch, self.nelec*3))
        self.pos.requires_grad = True

    def process_position(self):
        """Return the distance between electron and centers."""

        bas_coords = torch.zeros(self.nbas, 3)
        xyz = (self.pos.view(-1, self.nelec, 1, 3) -
               bas_coords[None, ...])
        r = torch.sqrt((xyz*xyz).sum(3))
        return xyz, r

    def test_value(self):
        xyz, r = self.process_position()
        self.harmonics(xyz, derivative=0)

    def test_grad(self):
        xyz, r = self.process_position()

        val_grad = self.harmonics(
            xyz, derivative=1, sum_grad=False)

        val = self.harmonics(xyz)
        val_grad_auto = grad(val, self.pos, torch.ones_like(val))[0]

        assert(torch.allclose(
            val_grad.sum(), val_grad_auto.sum(), atol=1E-6))

    def test_jac(self):
        xyz, r = self.process_position()
        val_jac = self.harmonics(xyz, derivative=1, sum_grad=True)
        val = self.harmonics(xyz)
        val_jac_auto = grad(val, self.pos, torch.ones_like(val))[0]

        assert(torch.allclose(
            val_jac.sum(), val_jac_auto.sum(), atol=1E-6))

    def test_lap(self):
        xyz, r = self.process_position()
        val_hess = self.harmonics(xyz, derivative=2)
        val = self.harmonics(xyz)
        val_hess_auto = hess(val, self.pos)

        assert(torch.allclose(
            val_hess.sum(), val_hess_auto.sum(), atol=1E-6))

    def test_mixed_der(self):
        xyz, r = self.process_position()
        val_hess = self.harmonics(xyz, derivative=3)
        val = self.harmonics(xyz)
        val_hess_auto = hess_mixed_terms(val, self.pos)

        assert(torch.allclose(
            val_hess.sum(), val_hess_auto.sum(), atol=1E-6))


if __name__ == "__main__":
    unittest.main()
