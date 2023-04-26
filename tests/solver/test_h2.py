import unittest

import numpy as np
import torch
import torch.optim as optim

from qmctorch.sampler import Hamiltonian, Metropolis
from qmctorch.solver import Solver
from qmctorch.utils import (plot_block, plot_blocking_energy,
                            plot_correlation_coefficient, plot_energy,
                            plot_integrated_autocorrelation_time,
                            plot_walkers_traj)
from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow

__PLOT__ = True


class TestH2(unittest.TestCase):

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
        self.wf = SlaterJastrow(self.mol, kinetic='auto',
                                configs='single(2,2)')

        # sampler
        self.sampler = Metropolis(
            nwalkers=1000,
            nstep=2000,
            step_size=0.5,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain('normal'),
            move={
                'type': 'all-elec',
                'proba': 'normal'})

        self.hmc_sampler = Hamiltonian(
            nwalkers=100,
            nstep=200,
            step_size=0.1,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain('normal'))

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        # solver
        self.solver = Solver(wf=self.wf, sampler=self.sampler,
                                          optimizer=self.opt)

        # ground state energy
        self.ground_state_energy = -1.16

        # ground state pos
        self.ground_state_pos = 0.69

    def test1_single_point(self):

        self.solver.wf.ao.atom_coords[0, 2] = -self.ground_state_pos
        self.solver.wf.ao.atom_coords[1, 2] = self.ground_state_pos
        self.solver.sampler = self.sampler

        # sample and compute observables
        obs = self.solver.single_point()
        e, v = obs.energy, obs.variance

        # values on different arch
        expected_energy = [-1.1464850902557373,
                           -1.14937478612449]

        # values on different arch
        expected_variance = [0.9279592633247375,
                             0.7445300449383236]

        assert(np.any(np.isclose(e.data.item(), np.array(expected_energy))))
        assert(np.any(np.isclose(v.data.item(), np.array(expected_variance))))

    def test2_single_point_hmc(self):

        self.solver.wf.ao.atom_coords[0, 2] = -self.ground_state_pos
        self.solver.wf.ao.atom_coords[1, 2] = self.ground_state_pos
        self.solver.sampler = self.hmc_sampler

        # sample and compute observables
        obs = self.solver.single_point()
        e, v = obs.energy, obs.variance

        # values on different arch
        expected_energy = [-1.0877732038497925,
                           -1.088576]

        # values on different arch
        expected_variance = [0.14341972768306732,
                             0.163771]

        assert(np.any(np.isclose(e.data.item(), np.array(expected_energy))))
        assert(np.any(np.isclose(v.data.item(), np.array(expected_variance))))

    def test3_wf_opt(self):
        self.solver.sampler = self.sampler

        self.solver.configure(track=['local_energy', 'parameters'],
                              loss='energy', grad='auto')
        obs = self.solver.run(5)
        if __PLOT__:
            plot_energy(obs.local_energy, e0=-
                        1.1645, show_variance=True)

    def test4_geo_opt(self):

        self.solver.wf.ao.atom_coords[0,
                                      2].data = torch.as_tensor(-0.37)
        self.solver.wf.ao.atom_coords[1,
                                      2].data = torch.as_tensor(0.37)

        self.solver.configure(track=['local_energy'],
                              loss='energy', grad='auto')
        self.solver.geo_opt(5, nepoch_wf_init=10, nepoch_wf_update=5)

        # load the best model
        self.solver.wf.load(self.solver.hdf5file, 'geo_opt')
        self.solver.wf.eval()

        # sample and compute variables
        obs = self.solver.single_point()
        e, v = obs.energy, obs.variance

        e = e.data.numpy()
        v = v.data.numpy()

        # it might be too much to assert with the ground state energy
        assert(e > 2 * self.ground_state_energy and e < 0.)
        assert(v > 0 and v < 2.)

    def test5_sampling_traj(self):
        self.solver.sampler = self.sampler

        self.solver.sampler.nstep = 100
        self.solver.sampler.ntherm = 0
        self.solver.sampler.ndecor = 1

        pos = self.solver.sampler(self.solver.wf.pdf)
        obs = self.solver.sampling_traj(pos)

        if __PLOT__:
            plot_walkers_traj(obs.local_energy)
            plot_block(obs.local_energy)

            plot_blocking_energy(obs.local_energy, block_size=10)
            plot_correlation_coefficient(obs.local_energy)
            plot_integrated_autocorrelation_time(obs.local_energy)


if __name__ == "__main__":
    # unittest.main()
    t = TestH2()
    t.setUp()
    # t.test2_single_point_hmc()
    # t.test1_single_point()
    t.test3_wf_opt()
    # t.test5_sampling_traj()
