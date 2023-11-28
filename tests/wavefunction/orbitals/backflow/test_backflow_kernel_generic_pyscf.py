import unittest

import torch
from pyscf import gto
from torch import nn
from torch.autograd import Variable, grad
import numpy as np

from qmctorch.scf import Molecule
from qmctorch.wavefunction.orbitals.backflow.kernels import BackFlowKernelBase
from qmctorch.wavefunction.jastrows.distance.electron_electron_distance import (
    ElectronElectronDistance,
)

torch.set_default_tensor_type(torch.DoubleTensor)

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


class GenericBackFlowKernel(BackFlowKernelBase):
    def __init__(self, mol, cuda=False):
        """Define a generic kernel to test the auto diff features."""
        super().__init__(mol, cuda)
        eps = 1e-4
        self.weight = nn.Parameter(eps * torch.rand(self.nelec, self.nelec)).to(
            self.device
        )

    def _backflow_kernel(self, ree):
        """Computes the backflow kernel:

        .. math:
            \\eta(r_{ij}) = w_{ij} r_{ij}^2

        Args:
            r (torch.tensor): e-e distance Nbatch x Nelec x Nelec

        Returns:
            torch.tensor : f(r) Nbatch x Nelec x Nelec
        """
        return self.weight * ree * ree


class TestGenericBackFlowKernel(unittest.TestCase):
    def setUp(self):
        # define the molecule
        at = "C 0 0 0"
        basis = "dzp"
        self.mol = Molecule(atom=at, calculator="pyscf", basis=basis, unit="bohr")

        # define the kernel
        self.kernel = GenericBackFlowKernel(self.mol)
        self.edist = ElectronElectronDistance(self.mol.nelec)

        # define the grid points
        self.npts = 11
        self.pos = torch.rand(self.npts, self.mol.nelec * 3)
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True

    def test_derivative_backflow_kernel(self):
        """Test the derivative of the kernel function
        wrt the elec-elec distance."""

        ree = self.edist(self.pos)
        bf_kernel = self.kernel(ree)
        dbf_kernel_auto = grad(bf_kernel, ree, grad_outputs=torch.ones_like(bf_kernel))[
            0
        ]
        dbf_kernel = self.kernel(ree, derivative=1)

        assert torch.allclose(dbf_kernel.sum(), dbf_kernel_auto.sum())
        assert torch.allclose(dbf_kernel, dbf_kernel_auto)

    def test_second_derivative_backflow_kernel(self):
        """Test the 2nd derivative of the kernel function
        wrt the elec-elec distance."""

        ree = self.edist(self.pos)
        bf_kernel = self.kernel(ree)

        d2bf_kernel_auto = hess_single_element(bf_kernel, ree)

        d2bf_kernel = self.kernel(ree, derivative=2)

        assert torch.allclose(d2bf_kernel.sum(), d2bf_kernel_auto.sum())
        assert torch.allclose(d2bf_kernel, d2bf_kernel_auto)

    def test_derivative_backflow_kernel_pos(self):
        """Test the derivative of the kenel function wrt the pos of the elecs.
        Note that the derivative edist(pos,1) returns d r_ij = d/dx_i r_ij
        and that d/dx_j r_ij = d/d_xi r_ij = - d/dx_i r_ji
        i.e. edist(pos,1) returns half of the derivatives

        so to obatin the same values than autograd we need to double d/dx_i r_ij
        """

        # compute the ee dist
        ree = self.edist(self.pos)

        # compute the kernel values
        bfpos = self.kernel(ree)

        # computes the derivative of the ee dist
        di_ree = self.edist(self.pos, 1)
        dj_ree = di_ree

        # compute the derivative of the kernal values
        bf_der = self.kernel(ree, derivative=1)

        # get the der of the bf wrt the first elec in ree
        di_bfpos = bf_der.unsqueeze(1) * di_ree

        # need to take the transpose here
        # get the der of the bf wrt the second elec in ree
        dj_bfpos = (bf_der.permute(0, 2, 1)).unsqueeze(1) * dj_ree

        # add both components
        d_bfpos = di_bfpos + dj_bfpos

        # computes the the derivative of the kernal values with autograd
        dbfpos_grad = grad(bfpos, self.pos, grad_outputs=torch.ones_like(bfpos))[0]

        # checksum
        assert torch.allclose(d_bfpos.sum(), dbfpos_grad.sum())

        # reshape and check individual elements
        dbfpos = d_bfpos.sum(-1).permute(0, 2, 1).reshape(self.npts, -1)
        assert torch.allclose(dbfpos, dbfpos_grad)

    def test_second_derivative_backflow_kernel_pos(self):
        """Test the derivative of the kenel function wrt the pos of the elecs.
        Note that the derivative edist(pos,1) returns d r_ij = d/dx_i r_ij
        and that d/dx_j r_ij = d/d_xi r_ij = - d/dx_i r_ji
        i.e. edist(pos,1) returns half of the derivatives
        Same thing for edist(pos,2)

        so to obatin the same values than autograd we need to double d/dx_i r_ij
        """

        # compute the ee dist
        ree = self.edist(self.pos)

        # compute the kernel values
        bf_kernel = self.kernel(ree)

        # computes the derivative of the ee dist
        di_ree = self.edist(self.pos, 1)
        dj_ree = di_ree

        # computes the derivative of the ee dist
        d2i_ree = self.edist(self.pos, 2)
        d2j_ree = d2i_ree

        # compute the derivative of the kernel values
        d2bf_kernel = self.kernel(ree, derivative=2).unsqueeze(1) * di_ree * di_ree

        d2bf_kernel += (
            self.kernel(ree, derivative=2).permute(0, 2, 1).unsqueeze(1)
            * dj_ree
            * dj_ree
        )

        d2bf_kernel += self.kernel(ree, derivative=1).unsqueeze(1) * d2i_ree

        d2bf_kernel += (
            self.kernel(ree, derivative=1).permute(0, 2, 1).unsqueeze(1) * d2j_ree
        )

        # computes the the derivative of the kernal values with autograd
        d2bf_kernel_auto = hess(bf_kernel, self.pos)

        # checksum
        assert torch.allclose(d2bf_kernel.sum(), d2bf_kernel_auto.sum())

        # reshape and check individual elements
        d2bf_kernel = d2bf_kernel.sum(-1).permute(0, 2, 1).reshape(self.npts, -1)

        assert torch.allclose(d2bf_kernel, d2bf_kernel_auto)


if __name__ == "__main__":
    unittest.main()
