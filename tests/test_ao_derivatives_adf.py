import torch
from torch.autograd import Variable, grad, gradcheck
from qmctorch.wavefunction import Orbital, Molecule
from pyscf import gto

import numpy as np
import unittest

import matplotlib.pyplot as plt

import os

torch.set_default_tensor_type(torch.DoubleTensor)


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


class TestAOderivativesADF(unittest.TestCase):

    def setUp(self):

        # define the molecule
        self.mol = Molecule(load='hdf5/C_adf_dzp.hdf5')

        # define the wave function
        self.wf = Orbital(self.mol, include_all_mo=True)

        # define the grid points
        npts = 11
        self.pos = torch.rand(npts, self.mol.nelec * 3)
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True

    def test_ao_deriv(self):

        ao = self.wf.ao(self.pos)
        dao = self.wf.ao(self.pos, derivative=1)

        dao_grad = grad(
            ao, self.pos, grad_outputs=torch.ones_like(ao))[0]

        gradcheck(self.wf.ao, self.pos)
        assert(torch.allclose(dao.sum(), dao_grad.sum()))

    def test_ao_hess(self):

        ao = self.wf.ao(self.pos)
        d2ao = self.wf.ao(self.pos, derivative=2)
        d2ao_grad = hess(ao, self.pos)
        assert(torch.allclose(d2ao.sum(), d2ao_grad.sum()))


if __name__ == "__main__":
    # unittest.main()

    t = TestAOderivativesADF()
    t.setUp()
    t.test_ao_deriv()
    t.test_ao_hess()
