import unittest

import numpy as np
import torch
import torch.optim as optim

from .test_base_solver import BaseTestSolvers

from qmctorch.sampler import Hamiltonian
from qmctorch.solver import Solver

from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel

__PLOT__ = True


class TestH2SamplerHMC(BaseTestSolvers.BaseTestSolverMolecule):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

        # optimal parameters
        self.opt_r = 0.69  # the two h are at +0.69 and -0.69
        self.opt_sigma = 1.24

        # molecule
        self.mol = Molecule(
            atom="H 0 0 -0.69; H 0 0 0.69",
            unit="bohr",
            calculator="pyscf",
            basis="sto-3g",
        )

        # jastrow
        jastrow = JastrowFactor(self.mol, PadeJastrowKernel)

        # wave function
        self.wf = SlaterJastrow(
            self.mol, kinetic="jacobi", configs="single(2,2)", jastrow=jastrow
        )

        self.sampler = Hamiltonian(
            nwalkers=100,
            nstep=200,
            step_size=0.1,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain("normal"),
        )

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        # solver
        self.solver = Solver(wf=self.wf, sampler=self.sampler, optimizer=self.opt)

        # values on different arch
        self.expected_energy = [-1.0877732038497925, -1.088576]

        # values on different arch
        self.expected_variance = [0.14341972768306732, 0.163771]


if __name__ == "__main__":
    unittest.main()
