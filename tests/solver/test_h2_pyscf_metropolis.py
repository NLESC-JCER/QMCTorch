import unittest

import numpy as np
import torch
import torch.optim as optim

from .test_base_solver import BaseTestSolvers

from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver

from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel

__PLOT__ = True


class TestH2SamplerMH(BaseTestSolvers.BaseTestSolverMolecule):
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

        # values on different arch
        self.expected_energy = [-1.1464850902557373, -1.14937478612449]

        # values on different arch
        self.expected_variance = [0.9279592633247375, 0.7445300449383236]


if __name__ == "__main__":
    unittest.main()
    t = TestH2SamplerMH()
    t.setUp()
    # # t.test2_single_point_hmc()
    t.test1_single_point()
    # t.test3_wf_opt()
    # # t.test5_sampling_traj()
