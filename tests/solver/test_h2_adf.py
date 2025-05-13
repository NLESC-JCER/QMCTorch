from qmctorch.wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.scf import Molecule
from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver

import unittest
import torch
import torch.optim as optim

from .test_base_solver import BaseTestSolvers
from ..path_utils import PATH_TEST


class TestH2ADF(BaseTestSolvers.BaseTestSolverMolecule):
    def setUp(self):
        torch.manual_seed(0)

        # molecule
        path_hdf5 = (PATH_TEST / "hdf5/H2_adf_dzp.hdf5").absolute().as_posix()
        self.mol = Molecule(load=path_hdf5)

        # jastrow
        jastrow = JastrowFactor(self.mol, PadeJastrowKernel)

        # wave function
        self.wf = SlaterJastrow(
            self.mol, kinetic="jacobi", configs="single(2,2)", jastrow=jastrow
        )

        # sampler
        self.sampler = Metropolis(
            nwalkers=1000,
            nstep=2000,
            step_size=0.5,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain("normal"),
            move={"type": "all-elec", "proba": "normal"},
        )

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        # solver
        self.solver = Solver(wf=self.wf, sampler=self.sampler, optimizer=self.opt)

        # vals on different archs
        self.expected_energy = [-1.1572532653808594, -1.1501641653648578]

        self.expected_variance = [0.05085879936814308, 0.05094174843043177]


if __name__ == "__main__":
    unittest.main()
    # t = TestH2ADF()
    # t.setUp()
    # t.test_single_point()
