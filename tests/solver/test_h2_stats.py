import unittest

import numpy as np
import torch
import torch.optim as optim

from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver
from qmctorch.utils import (plot_block, plot_blocking_energy,
                            plot_correlation_coefficient,
                            plot_integrated_autocorrelation_time,
                            plot_walkers_traj)
from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow


__PLOT__ = False


class TestH2Stat(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)
        np.random.seed(0)

        # optimal parameters
        self.opt_r = 0.69  # the two h are at +0.69 and -0.69
        self.opt_sigma = 1.24

        # molecule
        self.mol = Molecule(
            atom='H 0 0 -0.69; H 0 0 0.69',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g')

        # wave function
        self.wf = SlaterJastrow(self.mol, kinetic='jacobi',
                                configs='single(2,2)')

        # sampler
        self.sampler = Metropolis(
            nwalkers=100,
            nstep=500,
            step_size=0.5,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            ntherm=0,
            ndecor=1,
            init=self.mol.domain('normal'),
            move={
                'type': 'all-elec',
                'proba': 'normal'})

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        # solver
        self.solver = Solver(wf=self.wf, sampler=self.sampler,
                                          optimizer=self.opt)

    def test_sampling_traj(self):

        pos = self.solver.sampler(self.solver.wf.pdf)
        obs = self.solver.sampling_traj(pos)

        plot_walkers_traj(obs.local_energy)
        plot_block(obs.local_energy)

    def test_stat(self):

        pos = self.solver.sampler(self.solver.wf.pdf)
        obs = self.solver.sampling_traj(pos)

        if __PLOT__:
            plot_blocking_energy(obs.local_energy, block_size=10)
            plot_correlation_coefficient(obs.local_energy)
            plot_integrated_autocorrelation_time(obs.local_energy)


if __name__ == "__main__":
    # unittest.main()
    t = TestH2Stat()
    t.setUp()
    t.test_sampling_traj()
    t.test_stat()
    # t.test1_single_point()
    # t.test3_wf_opt()
