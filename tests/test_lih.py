import torch
import torch.optim as optim

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.solver import SolverOrbital
from qmctorch.sampler import Metropolis, Hamiltonian
from qmctorch.utils import plot_energy, plot_data, plot_walkers_traj, plot_block

import platform

import numpy as np
import unittest


class TestLiH(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)
        np.random.seed(0)

        # molecule
        self.mol = Molecule(
            atom='Li 0 0 0; H 0 0 3.015',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g')

        # wave function
        self.wf = Orbital(self.mol, kinetic='jacobi',
                          configs='single(2,2)',
                          use_jastrow=True,
                          include_all_mo=False)

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
        self.solver = SolverOrbital(wf=self.wf, sampler=self.sampler,
                                    optimizer=self.opt)

    def test1_single_point(self):

        # sample and compute observables
        obs = self.solver.single_point()
        e, v = obs.energy, obs.variance

        # # values on different arch
        # expected_energy = [-1.1464850902557373,
        #                    -1.14937478612449]

        # # values on different arch
        # expected_variance = [0.9279592633247375,
        #                      0.7445300449383236]

        # assert(np.any(np.isclose(e.data.item(), np.array(expected_energy))))
        # assert(np.any(np.isclose(v.data.item(), np.array(expected_variance))))

    def test2_wf_opt_grad_auto(self):
        self.solver.sampler = self.sampler

        self.solver.configure(task='wf_opt')
        self.solver.track_observable(['local_energy'])
        obs = self.solver.run(5, loss='energy', grad='auto')

    def test3_wf_opt_grad_manual(self):
        self.solver.sampler = self.sampler

        self.solver.configure(task='wf_opt')
        self.solver.track_observable(['local_energy'])
        obs = self.solver.run(5, loss='energy', grad='manual')


if __name__ == "__main__":
    unittest.main()
    t = TestLiH()
    t.setUp()
    t.test1_single_point()
    t.test2_wf_opt_grad_auto()
    t.test3_wf_opt_grad_manual()
