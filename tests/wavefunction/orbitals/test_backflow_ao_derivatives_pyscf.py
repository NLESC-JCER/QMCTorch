import unittest

import torch
from pyscf import gto
from torch.autograd import Variable, grad, gradcheck
import numpy as np
from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow
from qmctorch.wavefunction.orbitals.atomic_orbitals_backflow import AtomicOrbitalsBackFlow
from qmctorch.wavefunction.orbitals.backflow.backflow_kernel_inverse import BackFlowKernelnInverse
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
            self.mol, BackFlowKernelnInverse)

        # define the grid points
        self.npts = 11
        self.pos = torch.rand(self.npts, self.mol.nelec * 3)
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True

    def test_derivative_backflow_kernel(self):
        """Test the derivative of the kernel function
            wrt the elec-elec distance."""

        ree = self.ao.backflow_trans.edist(self.pos)
        bf_kernel = self.ao.backflow_trans.backflow_kernel(ree)
        dbf_kernel_auto = grad(
            bf_kernel, ree, grad_outputs=torch.ones_like(bf_kernel))[0]
        dbf_kernel = self.ao.backflow_trans.backflow_kernel(
            ree, derivative=1)

        assert(torch.allclose(dbf_kernel.sum(), dbf_kernel_auto.sum()))
        assert(torch.allclose(dbf_kernel, dbf_kernel_auto))

    def test_second_derivative_backflow_kernel(self):
        """Test the 2nd derivative of the kernel function
           wrt the elec-elec distance."""

        ree = self.ao.backflow_trans.edist(self.pos)
        bf_kernel = self.ao.backflow_trans.backflow_kernel(ree)

        d2bf_kernel_auto = hess_single_element(bf_kernel, ree)

        d2bf_kernel = self.ao.backflow_trans.backflow_kernel(
            ree, derivative=2)

        assert(torch.allclose(d2bf_kernel.sum(), d2bf_kernel_auto.sum()))
        assert(torch.allclose(d2bf_kernel, d2bf_kernel_auto))

    def test_derivative_backflow_kernel_pos(self):
        """Test the derivative of the kenel function wrt the pos of the elecs.
        Note that the derivative edist(pos,1) returns d r_ij = d/dx_i r_ij
        and that d/dx_j r_ij = d/d_xi r_ij = - d/dx_i r_ji
        i.e. edist(pos,1) returns half of the derivatives

        so to obatin the same values than autograd we need to double d/dx_i r_ij
        """

        # compute the ee dist
        ree = self.ao.backflow_trans.edist(self.pos)

        # compute the kernel values
        bfpos = self.ao.backflow_trans.backflow_kernel(ree)

        # computes the derivative of the ee dist
        di_ree = self.ao.backflow_trans.edist(self.pos, 1)
        dj_ree = di_ree

        # compute the derivative of the kernal values
        bf_der = self.ao.backflow_trans.backflow_kernel(
            ree, derivative=1)

        # get the der of the bf wrt the first elec in ree
        di_bfpos = bf_der.unsqueeze(1) * di_ree

        # need to take the transpose here
        # get the der of the bf wrt the second elec in ree
        dj_bfpos = (bf_der.permute(0, 2, 1)).unsqueeze(1) * dj_ree

        # add both components
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

    def test_second_derivative_backflow_kernel_pos(self):
        """Test the derivative of the kenel function wrt the pos of the elecs.
        Note that the derivative edist(pos,1) returns d r_ij = d/dx_i r_ij
        and that d/dx_j r_ij = d/d_xi r_ij = - d/dx_i r_ji
        i.e. edist(pos,1) returns half of the derivatives
        Same thing for edist(pos,2)

        so to obatin the same values than autograd we need to double d/dx_i r_ij
        """

        # compute the ee dist
        ree = self.ao.backflow_trans.edist(self.pos)

        # compute the kernel values
        bf_kernel = self.ao.backflow_trans.backflow_kernel(ree)

        # computes the derivative of the ee dist
        di_ree = self.ao.backflow_trans.edist(self.pos, 1)
        dj_ree = di_ree

        # computes the derivative of the ee dist
        d2i_ree = self.ao.backflow_trans.edist(self.pos, 2)
        d2j_ree = d2i_ree

        # compute the derivative of the kernel values
        d2bf_kernel = self.ao.backflow_trans.backflow_kernel(
            ree, derivative=2).unsqueeze(1) * di_ree * di_ree

        d2bf_kernel += self.ao.backflow_trans.backflow_kernel(
            ree, derivative=2).permute(0, 2, 1).unsqueeze(1) * dj_ree * dj_ree

        d2bf_kernel += self.ao.backflow_trans.backflow_kernel(
            ree, derivative=1).unsqueeze(1) * d2i_ree

        d2bf_kernel += self.ao.backflow_trans.backflow_kernel(
            ree, derivative=1).permute(0, 2, 1).unsqueeze(1) * d2j_ree

        # computes the the derivative of the kernal values with autograd
        d2bf_kernel_auto = hess(bf_kernel, self.pos)

        # checksum
        assert(torch.allclose(d2bf_kernel.sum(), d2bf_kernel_auto.sum()))

        # reshape and check individual elements
        d2bf_kernel = d2bf_kernel.sum(-1).permute(0, 2,
                                                  1).reshape(self.npts, -1)

        assert(torch.allclose(d2bf_kernel, d2bf_kernel_auto))

    def test_backflow_derivative(self):
        """Test the derivative of the bf coordinate wrt the initial positions."""

        # compute backflow pos
        q = self.ao.backflow_trans(self.pos)

        # compute der of the backflow pos wrt the
        # original pos
        dq = self.ao.backflow_trans(self.pos, derivative=1)

        # compute der of the backflow pos wrt the
        # original pos using autograd
        dq_grad = grad(
            q, self.pos, grad_outputs=torch.ones_like(self.pos))[0]

        # checksum
        assert(torch.allclose(dq.sum(), dq_grad.sum()))

        # permute and check elements
        dq = dq.sum([1, 3])
        dq = dq.permute(0, 2, 1)

        dq_grad = dq_grad.reshape(self.npts, self.mol.nelec, 3)
        assert(torch.allclose(dq, dq_grad))

    def test_backflow_second_derivative(self):
        """Test the derivative of the bf coordinate wrt the initial positions."""

        # compute backflow pos
        q = self.ao.backflow_trans(self.pos)

        # compute der of the backflow pos wrt the
        # original pos
        d2q = self.ao.backflow_trans(self.pos, derivative=2)

        # compute der of the backflow pos wrt the
        # original pos using autograd
        d2q_auto = hess(q, self.pos)

        # checksum
        assert(torch.allclose(d2q.sum(), d2q_auto.sum()))

        # permute and check elements
        d2q = d2q.sum([1, 3])
        d2q = d2q.permute(0, 2, 1)
        d2q_auto = d2q_auto.reshape(self.npts, self.mol.nelec, 3)

        assert(torch.allclose(d2q, d2q_auto))

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
        print(d2ao.sum(), d2ao_grad.sum())
        assert(torch.allclose(d2ao.sum(), d2ao_grad.sum()))

        print(d2ao.shape, d2ao_grad.shape)
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
    t = TestBFAOderivativesPyscf()
    t.setUp()

    t.test_derivative_backflow_kernel()
    t.test_derivative_backflow_kernel_pos()
    t.test_backflow_derivative()

    t.test_second_derivative_backflow_kernel()
    t.test_second_derivative_backflow_kernel_pos()
    t.test_backflow_second_derivative()

    t.test_ao_gradian()
    t.test_ao_jacobian()
    t.test_ao_hess()

    # t.test_all_ao_values()

    # unittest.main()
