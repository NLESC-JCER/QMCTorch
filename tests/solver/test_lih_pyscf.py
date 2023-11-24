import unittest

import numpy as np
import torch
import torch.optim as optim

from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver
from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel

from .test_base_solver import BaseTestSolvers


class TestLiH(BaseTestSolvers.BaseTestSolverMolecule):

    def setUp(self):

        torch.manual_seed(0)
        np.random.seed(0)

        # molecule
        self.mol = Molecule(
            atom='Li 0 0 0; H 0 0 3.015',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g')

        # jastrow
        jastrow = JastrowFactor(self.mol, PadeJastrowKernel)

        # wave function
        self.wf = SlaterJastrow(self.mol, kinetic='jacobi',
                                configs='single(2,2)',
                                include_all_mo=False, jastrow=jastrow)

        # sampler
        self.sampler = Metropolis(
            nwalkers=500,
            nstep=200,
            step_size=0.05,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain('normal'),
            move={
                'type': 'all-elec',
                'proba': 'normal'})

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        # solver
        self.solver = Solver(wf=self.wf, sampler=self.sampler,
                                          optimizer=self.opt)


if __name__ == "__main__":
    unittest.main()
    # t = TestLiH()
    # t.setUp()
    # t.test1_single_point()
    # t.test2_wf_opt_grad_auto()
    # t.test3_wf_opt_grad_manual()