import unittest

import torch
from torch.autograd import Variable, grad
import numpy as np

from qmctorch.scf import Molecule
from qmctorch.wavefunction.orbitals.backflow.kernels import BackFlowKernelExp
from qmctorch.wavefunction.jastrows.distance.electron_electron_distance import (
    ElectronElectronDistance,
)
from qmctorch.utils import set_torch_double_precision
from .test_backflow_base import BaseTestCases

set_torch_double_precision()

torch.manual_seed(101)
np.random.seed(101)


class TestBackFlowKernel(BaseTestCases.TestBackFlowKernelBase):
    def setUp(self):
        # define the molecule
        at = "C 0 0 0"
        basis = "dzp"
        self.mol = Molecule(atom=at, calculator="pyscf", basis=basis, unit="bohr")

        # define the kernel
        self.kernel = BackFlowKernelExp(self.mol)
        self.edist = ElectronElectronDistance(self.mol.nelec)

        # define the grid points
        self.npts = 11
        self.pos = torch.rand(self.npts, self.mol.nelec * 3)
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True


if __name__ == "__main__":
    unittest.main()
