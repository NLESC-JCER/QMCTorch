from ...path_utils import PATH_TEST
import unittest
import torch

from .base_test_ao import BaseTestAO

from torch.autograd import Variable

from qmctorch.scf import Molecule
from qmctorch.wavefunction.orbitals.atomic_orbitals import AtomicOrbitals

torch.set_default_tensor_type(torch.DoubleTensor)

torch.set_default_tensor_type(torch.DoubleTensor)


class TestAOderivativesADF(BaseTestAO.BaseTestAOderivatives):
    def setUp(self):
        # define the molecule
        path_hdf5 = PATH_TEST / "hdf5/C_adf_dzp.hdf5"
        self.mol = Molecule(load=path_hdf5)

        # define the wave function
        self.ao = AtomicOrbitals(self.mol)

        # define the grid points
        npts = 11
        self.pos = torch.rand(npts, self.mol.nelec * 3)
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True


if __name__ == "__main__":
    unittest.main()
    # t = TestAOderivativesADF()
    # t.setUp()
    # t.test_ao_deriv()
    # t.test_ao_hess()
