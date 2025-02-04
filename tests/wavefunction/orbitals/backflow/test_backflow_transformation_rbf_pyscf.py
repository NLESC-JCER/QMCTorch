import unittest

import torch
from torch.autograd import Variable, grad
import numpy as np
from qmctorch.scf import Molecule
from qmctorch.wavefunction.orbitals.backflow.backflow_transformation import (
    BackFlowTransformation,
)
from qmctorch.wavefunction.orbitals.backflow.kernels import BackFlowKernelRBF
from qmctorch.utils import set_torch_double_precision
set_torch_double_precision()

torch.manual_seed(101)
np.random.seed(101)


def hess(out, pos):
    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos, grad_outputs=z, only_inputs=True, create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[1]):
        tmp = grad(
            jacob[:, idim], pos, grad_outputs=z, only_inputs=True, create_graph=True
        )[0]

        hess[:, idim] = tmp[:, idim]

    return hess


def hess_single_element(out, inp):
    shape = out.shape
    out = out.reshape(-1, 1)

    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, inp, grad_outputs=z, only_inputs=True, create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape))

    hess = grad(jacob, inp, grad_outputs=z, only_inputs=True, create_graph=True)[0]

    return hess.reshape(*shape)


class TestBackFlowTransformation(unittest.TestCase):
    def setUp(self):
        # define the molecule
        at = "C 0 0 0"
        basis = "dzp"
        self.mol = Molecule(atom=at, calculator="pyscf", basis=basis, unit="bohr")

        # define the backflow transformation
        self.backflow_trans = BackFlowTransformation(self.mol, BackFlowKernelRBF)

        # define the grid points
        self.npts = 11
        self.pos = torch.rand(self.npts, self.mol.nelec * 3)
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True

    def test_backflow_derivative(self):
        """Test the derivative of the bf coordinate wrt the initial positions."""

        # compute backflow pos
        q = self.backflow_trans(self.pos)

        # compute der of the backflow pos wrt the
        # original pos
        dq = self.backflow_trans(self.pos, derivative=1).squeeze()

        # compute der of the backflow pos wrt the
        # original pos using autograd
        dq_grad = grad(q, self.pos, grad_outputs=torch.ones_like(self.pos))[0]

        # checksum
        assert torch.allclose(dq.sum(), dq_grad.sum())

        # permute and check elements
        dq = dq.sum([1, 3])
        dq = dq.permute(0, 2, 1)

        dq_grad = dq_grad.reshape(self.npts, self.mol.nelec, 3)
        assert torch.allclose(dq, dq_grad)

    def test_backflow_second_derivative(self):
        """Test the derivative of the bf coordinate wrt the initial positions."""

        # compute backflow pos
        q = self.backflow_trans(self.pos)

        # compute der of the backflow pos wrt the
        # original pos
        d2q = self.backflow_trans(self.pos, derivative=2).squeeze()

        # compute der of the backflow pos wrt the
        # original pos using autograd
        d2q_auto = hess(q, self.pos)

        # checksum
        assert torch.allclose(d2q.sum(), d2q_auto.sum())

        # permute and check elements
        d2q = d2q.sum([1, 3])
        d2q = d2q.permute(0, 2, 1)
        d2q_auto = d2q_auto.reshape(self.npts, self.mol.nelec, 3)

        assert torch.allclose(d2q, d2q_auto)


if __name__ == "__main__":
    unittest.main()
