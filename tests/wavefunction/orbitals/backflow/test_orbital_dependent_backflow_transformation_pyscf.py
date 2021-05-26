import unittest

import torch
from pyscf import gto
from torch.autograd import Variable, grad, gradcheck
import numpy as np
from qmctorch.scf import Molecule
from qmctorch.wavefunction.orbitals.backflow.orbital_dependent_backflow_transformation import OrbitalDependentBackFlowTransformation
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


class TestOrbitalDependentBackFlowTransformation(unittest.TestCase):

    def setUp(self):

        # define the molecule
        at = 'C 0 0 0'
        basis = 'dzp'
        self.mol = Molecule(atom=at,
                            calculator='pyscf',
                            basis=basis,
                            unit='bohr')

        # define the backflow transformation
        self.backflow_trans = OrbitalDependentBackFlowTransformation(
            self.mol, BackFlowKernelInverse)

        # set the weights to random
        for ker in self.backflow_trans.backflow_kernel.orbital_dependent_kernel:
            ker.weight.data[0] = torch.rand(1)

        # define the grid points
        self.npts = 11
        self.pos = torch.rand(self.npts, self.mol.nelec * 3)
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True

    def test_backflow_derivative(self):
        """Test the derivative of the bf coordinate wrt the initial positions."""

        # compute backflow pos
        q = self.backflow_trans(self.pos)
        nao = q.shape[1]
        # compute der of the backflow pos wrt the
        # original pos
        dq = self.backflow_trans(self.pos, derivative=1)

        # compute der of the backflow pos wrt the
        # original pos using autograd
        dq_grad = None
        for iq in range(nao):
            qao = q[:, iq, ...]
            dqao = grad(
                qao, self.pos, grad_outputs=torch.ones_like(self.pos), retain_graph=True)[0]
            if dq_grad is None:
                dq_grad = dqao
            else:
                dq_grad = torch.cat(
                    (dq_grad, dqao), axis=self.backflow_trans.backflow_kernel.stack_axis)
        # checksum
        assert(torch.allclose(dq.sum(), dq_grad.sum()))

        # permute and check elements
        dq = dq.sum([2, 4])
        dq = dq.permute(0, 1, 3, 2)
        dq_grad = dq_grad.reshape(self.npts, nao, self.mol.nelec, 3)

        assert(torch.allclose(dq, dq_grad))

    def test_backflow_second_derivative(self):
        """Test the derivative of the bf coordinate wrt the initial positions."""

        # compute backflow pos
        q = self.backflow_trans(self.pos)
        nao = q.shape[1]

        # compute der of the backflow pos wrt the
        # original pos
        d2q = self.backflow_trans(self.pos, derivative=2)

        # compute der of the backflow pos wrt the
        # original pos using autograd
        d2q_auto = None
        for iq in range(nao):
            qao = q[:, iq, ...]
            d2qao = hess(qao, self.pos)
            if d2q_auto is None:
                d2q_auto = d2qao
            else:
                d2q_auto = torch.cat(
                    (d2q_auto, d2qao), axis=self.backflow_trans.backflow_kernel.stack_axis)

        # checksum
        assert(torch.allclose(d2q.sum(), d2q_auto.sum()))

        # permute and check elements
        d2q = d2q.sum([2, 4])
        d2q = d2q.permute(0, 1, 3, 2)
        d2q_auto = d2q_auto.reshape(
            self.npts, nao,  self.mol.nelec, 3)

        assert(torch.allclose(d2q, d2q_auto))


if __name__ == "__main__":
    unittest.main()
