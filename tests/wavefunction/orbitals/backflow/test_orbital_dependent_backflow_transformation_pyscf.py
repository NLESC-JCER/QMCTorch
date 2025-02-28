import unittest

import torch
from torch.autograd import Variable, grad
import numpy as np
from qmctorch.scf import Molecule
from qmctorch.wavefunction.orbitals.backflow.backflow_transformation import (
    BackFlowTransformation,
)
from qmctorch.wavefunction.orbitals.backflow.kernels import BackFlowKernelInverse
from qmctorch.utils import set_torch_double_precision
from .test_backflow_base import BaseTestCases

set_torch_double_precision()

torch.manual_seed(101)
np.random.seed(101)


class TestOrbitalDependentBackFlowTransformation(
    BaseTestCases.TestOrbitalDependentBackFlowTransformationBase
):
    def setUp(self):
        # define the molecule
        at = "C 0 0 0"
        basis = "dzp"
        self.mol = Molecule(atom=at, calculator="pyscf", basis=basis, unit="bohr")

        # define the backflow transformation
        self.backflow_trans = BackFlowTransformation(
            self.mol, BackFlowKernelInverse, orbital_dependent=True
        )

        # set the weights to random
        for ker in self.backflow_trans.backflow_kernel.orbital_dependent_kernel:
            ker.weight.data[0] = torch.rand(1)

        # define the grid points
        self.npts = 11
        self.pos = torch.rand(self.npts, self.mol.nelec * 3)
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True


if __name__ == "__main__":
    unittest.main()
