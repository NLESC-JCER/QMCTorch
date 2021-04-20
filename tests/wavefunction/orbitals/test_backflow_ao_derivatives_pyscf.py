import unittest

import torch
from pyscf import gto
from torch.autograd import Variable, grad, gradcheck

from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow
from qmctorch.wavefunction.orbitals.atomic_orbitals_backflow import AtomicOrbitalsBackFlow
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
        self.ao = AtomicOrbitalsBackFlow(self.mol)

        # define the grid points
        self.npts = 11
        self.pos = torch.rand(self.npts, self.mol.nelec * 3)
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True

    def test_backflow_kernel(self):
        """Test the derivative of the kernel function wrt the elec-elec distance."""

        ree = self.ao.edist(self.pos)
        bfpos = self.ao._backflow_kernel(ree)

        dbfpos_grad = grad(
            bfpos, ree, grad_outputs=torch.ones_like(bfpos))[0]

        dbfpos = self.ao._backflow_kernel_derivative(ree)
        assert(torch.allclose(dbfpos.sum(), dbfpos_grad.sum()))
        assert(torch.allclose(dbfpos, dbfpos_grad))

    def test_backflow_kernel_pos(self):
        """Test the derivative of the kenel function wrt the pos of the elecs.
        Note that the derivative edist(pos,1) returns drr_ij = d/dx_i r_ij
        and that d/dx_j r_ij = d/d_xi r_ij = - d/dx_i r_ji
        i.e. edist(pos,1) returns half of the derivatives

        so to obatin the same values than autograd we need to double d/dx_i r_ij
        """

        # compute the ee dist
        ree = self.ao.edist(self.pos)

        # compute the kernel values
        bfpos = self.ao._backflow_kernel(ree)

        # computes the derivative of the ee dist
        di_ree = self.ao.edist(self.pos, 1)
        dj_ree = di_ree

        # compute the derivative of the kernal values
        di_bfpos = self.ao._backflow_kernel_derivative(
            ree).unsqueeze(1) * di_ree

        dj_bfpos = self.ao._backflow_kernel_derivative(
            ree).unsqueeze(1) * dj_ree

        d_bfpos = di_bfpos + dj_bfpos

        # computes the the derivative of the kernal values with autograd
        dbfpos_grad = grad(
            bfpos, self.pos, grad_outputs=torch.ones_like(bfpos))[0]

        # checksum
        assert(torch.allclose(d_bfpos.sum(), dbfpos_grad.sum()))

        # reshape and check individual elements
        dbfpos = d_bfpos.sum(-1).permute(0, 2,
                                         1).reshape(self.npts, -1)
        assert(torch.allclose(dbfpos, dbfpos_grad))

    def test_backflow_derivative(self):
        """Test the derivative pf the bf coordinate wrt the initial positions."""

        # compute backflow pos
        q = self.ao._backflow(self.pos)

        # compute der of the backflow pos wrt the
        # original pos
        dq = self.ao._backflow_derivative(self.pos)

        # compute der of the backflow pos wrt the
        # original pos using autograd
        dq_grad = grad(
            q, self.pos, grad_outputs=torch.ones_like(self.pos))[0]

        # checksum
        assert(torch.allclose(dq.sum(), dq_grad.sum()))

        # permute and check elements
        dq = dq.sum(-1).permute(0, 2,
                                1).reshape(self.npts, self.mol.nelec*3)
        assert(torch.allclose(dq, dq_grad))

    def test_ao_gradian(self):
        """Test the calculation of the gradient of the at
        wrt the original coordinates."""

        ao = self.ao(self.pos)
        dao = self.ao(self.pos, derivative=1, jacobian=False)

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


if __name__ == "__main__":
    t = TestBFAOderivativesPyscf()
    t.setUp()
    t.test_backflow_kernel()
    t.test_backflow_kernel_pos()
    t.test_backflow_derivative()
    t.test_ao_gradian()
    t.test_ao_jacobian()
    # unittest.main()
