import unittest
import numpy as np
import torch
from torch.autograd import Variable
from qmctorch.scf import Molecule
from qmctorch.wavefunction.orbitals.atomic_orbitals import AtomicOrbitals
from qmctorch.utils import set_torch_double_precision
from .base_test_ao import BaseTestAO

set_torch_double_precision()


class TestAOderivativesPyscf(BaseTestAO.BaseTestAOderivatives):
    def setUp(self):
        torch.manual_seed(101)
        np.random.seed(101)

        # define the molecule
        at = "Li 0 0 0; H 0 0 1"
        basis = "dzp"
        self.mol = Molecule(atom=at, calculator="pyscf", basis=basis, unit="bohr")

        # define the aos
        self.ao = AtomicOrbitals(self.mol)

        # define the grid points
        npts = 11
        self.pos = torch.rand(npts, self.mol.nelec * 3)
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True


if __name__ == "__main__":
    unittest.main()

    # t = TestAOderivativesPyscf()
    # t.setUp()
    # t.test_ao_mixed_der()
    # t.test_ao_all()
    # t.test_ao_deriv()
    # t.test_ao_hess()
