import unittest
import numpy as np
import torch
from pyscf import gto
from torch.autograd import Variable, grad, gradcheck

from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow

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


class TestAOderivativesPyscf(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(101)
        np.random.seed(101)

        # define the molecule
        at = 'Li 0 0 0; H 0 0 1'
        basis = 'dzp'
        self.mol = Molecule(atom=at,
                            calculator='pyscf',
                            basis=basis,
                            unit='bohr')

        self.m = gto.M(atom=at, basis=basis, unit='bohr')

        # define the wave function
        self.wf = SlaterJastrow(self.mol, include_all_mo=True)

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

    def test_ao_grad_sum(self):

        ao = self.wf.ao(self.pos)
        dao_sum = self.wf.ao(self.pos, derivative=1, sum_grad=True)
        dao = self.wf.ao(self.pos, derivative=1, sum_grad=False)

        assert(torch.allclose(dao_sum, dao.sum(-1)))

    def test_ao_hess(self):

        ao = self.wf.ao(self.pos)
        d2ao = self.wf.ao(self.pos, derivative=2)
        d2ao_grad = hess(ao, self.pos)
        assert(torch.allclose(d2ao.sum(), d2ao_grad.sum()))

    def test_ao_hess_sum(self):

        ao = self.wf.ao(self.pos)
        d2ao_sum = self.wf.ao(self.pos, derivative=2, sum_hess=True)
        d2ao = self.wf.ao(self.pos, derivative=2, sum_hess=False)
        assert(torch.allclose(d2ao_sum, d2ao.sum(-1)))

    def test_ao_mixed_der(self):
        ao = self.wf.ao(self.pos)
        d2ao = self.wf.ao(self.pos, derivative=3)
        d2ao_auto = hess_mixed_terms(ao, self.pos)

        assert(torch.allclose(d2ao.sum(), d2ao_auto.sum()))

    def test_ao_all(self):
        ao = self.wf.ao(self.pos)
        dao = self.wf.ao(self.pos, derivative=1, sum_grad=False)
        d2ao = self.wf.ao(self.pos, derivative=2)
        ao_all, dao_all, d2ao_all = self.wf.ao(
            self.pos, derivative=[0, 1, 2])

        assert(torch.allclose(ao, ao_all))
        assert(torch.allclose(dao, dao_all))
        assert(torch.allclose(d2ao, d2ao_all))


if __name__ == "__main__":
    # unittest.main()

    t = TestAOderivativesPyscf()
    t.setUp()
    t.test_ao_mixed_der()
    # t.test_ao_all()
    # t.test_ao_deriv()
    # t.test_ao_hess()
