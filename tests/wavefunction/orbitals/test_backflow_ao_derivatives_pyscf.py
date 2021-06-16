import unittest

import torch
from pyscf import gto
from torch.autograd import Variable, grad, gradcheck
import numpy as np
from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow
from qmctorch.wavefunction.orbitals.atomic_orbitals_backflow import AtomicOrbitalsBackFlow
from qmctorch.wavefunction.orbitals.backflow.kernels import BackFlowKernelInverse
torch.set_default_tensor_type(torch.DoubleTensor)

torch.manual_seed(101)
np.random.seed(101)


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


def hess_single_element(out, inp):

    shape = out.shape
    out = out.reshape(-1, 1)

    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, inp,
                 grad_outputs=z,
                 only_inputs=True,
                 create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape))

    hess = grad(jacob, inp,
                grad_outputs=z,
                only_inputs=True,
                create_graph=True)[0]

    return hess.reshape(*shape)


class TestBFAOderivativesPyscf(unittest.TestCase):

    def setUp(self):

        # define the molecule
        at = 'C 0 0 0'
        basis = 'dzp'
        self.mol = Molecule(atom=at,
                            calculator='pyscf',
                            basis=basis,
                            unit='bohr')

        # define the wave function
        self.ao = AtomicOrbitalsBackFlow(
            self.mol, BackFlowKernelInverse)

        # define the grid points
        self.npts = 11
        self.pos = torch.rand(self.npts, self.mol.nelec * 3)
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True

    def test_ao_gradian(self):
        """Test the calculation of the gradient of the at
        wrt the original coordinates."""

        ao = self.ao(self.pos)
        dao = self.ao(self.pos, derivative=1, sum_grad=False)

        dao_grad = grad(
            ao, self.pos, grad_outputs=torch.ones_like(ao))[0]

        assert(torch.allclose(dao.sum(), dao_grad.sum()))

        dao = dao.sum(-1).sum(-1)
        dao_grad = dao_grad.T
        assert(torch.allclose(dao, dao_grad))

    def test_ao_jacobian(self):

        ao = self.ao(self.pos)
        dao = self.ao(self.pos, derivative=1)

        dao_grad = grad(
            ao, self.pos, grad_outputs=torch.ones_like(ao))[0]

        assert(torch.allclose(dao.sum(), dao_grad.sum()))

        dao = dao.sum(-1).sum(-1)
        dao_grad = dao_grad.reshape(-1, self.ao.nelec, 3).sum(-1)
        dao_grad = dao_grad.T
        assert(torch.allclose(dao, dao_grad))

    def test_ao_hess(self):

        ao = self.ao(self.pos)
        d2ao = self.ao(self.pos, derivative=2)

        d2ao_grad = hess(ao, self.pos)
        assert(torch.allclose(d2ao.sum(), d2ao_grad.sum()))

        d2ao = d2ao.sum(-1).sum(-1)
        d2ao_grad = d2ao_grad.reshape(-1, self.ao.nelec, 3).sum(-1)
        d2ao_grad = d2ao_grad.T
        assert(torch.allclose(d2ao, d2ao_grad))

    def test_all_ao_values(self):
        ao = self.ao(self.pos)
        dao = self.ao(self.pos, derivative=1, sum_grad=False)
        d2ao = self.ao(self.pos, derivative=2, sum_hess=False)
        ao_all, dao_all, d2ao_all = self.ao(
            self.pos, derivative=[0, 1, 2])

        assert(torch.allclose(ao, ao_all))
        assert(torch.allclose(dao, dao_all))
        assert(torch.allclose(d2ao, d2ao_all))


if __name__ == "__main__":
    unittest.main()
