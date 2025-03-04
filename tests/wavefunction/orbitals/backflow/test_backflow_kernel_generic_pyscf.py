import unittest

import torch
from torch import nn
from torch.autograd import Variable, grad
import numpy as np

from qmctorch.scf import Molecule
from qmctorch.wavefunction.orbitals.backflow.kernels import BackFlowKernelBase
from qmctorch.wavefunction.jastrows.distance.electron_electron_distance import (
    ElectronElectronDistance,
)
from .test_backflow_base import BaseTestCases
from qmctorch.utils import set_torch_double_precision

set_torch_double_precision()

torch.manual_seed(101)
np.random.seed(101)


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


class TestGenericBackFlowKernel(BaseTestCases.TestBackFlowKernelBase):
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


if __name__ == "__main__":
    unittest.main()
