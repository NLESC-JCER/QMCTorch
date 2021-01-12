import unittest

import numpy as np
import torch
import torch.optim as optim

from qmctorch.sampler import Hamiltonian, Metropolis
from qmctorch.solver import SolverOrbital
from qmctorch.utils import (plot_block, plot_blocking_energy,
                            plot_correlation_coefficient, plot_energy,
                            plot_integrated_autocorrelation_time,
                            plot_walkers_traj)

from qmctorch.wavefunction.jastrows.fully_connected_jastrow import FullyConnectedJastrow

from qmctorch.scf import Molecule
from qmctorch.wavefunction import CorrelatedOrbital
from qmctorch.utils import set_torch_double_precision

__PLOT__ = False


class TestH2Correlated(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)
        np.random.seed(0)
        set_torch_double_precision()

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
        self.wf = CorrelatedOrbital(self.mol,
                                    kinetic='auto',
                                    configs='single_double(2,2)',
                                    jastrow_type=FullyConnectedJastrow)

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

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        # solver
        self.solver = SolverOrbital(wf=self.wf, sampler=self.sampler,
                                    optimizer=self.opt)

        # ground state energy
        self.ground_state_energy = -1.16

        # ground state pos
        self.ground_state_pos = 0.69

    def test_0_wavefunction(self):

        # artificial pos
        self.nbatch = 10
        self.pos = torch.tensor(np.random.rand(
            self.nbatch, self.wf.nelec*3))
        self.pos.requires_grad = True

        eauto = self.wf.kinetic_energy_autograd(self.pos)
        ejac = self.wf.kinetic_energy_jacobi(self.pos)
        print(torch.stack([eauto, ejac], axis=1).squeeze())
        assert torch.allclose(
            eauto.data, ejac.data, rtol=1E-4, atol=1E-4)

    def test1_single_point(self):

        self.solver.wf.ao.atom_coords[0, 2] = -self.ground_state_pos
        self.solver.wf.ao.atom_coords[1, 2] = self.ground_state_pos
        self.solver.sampler = self.sampler

        # sample and compute observables
        obs = self.solver.single_point()
        e, v = obs.energy, obs.variance

        # values on different arch
        expected_energy = [-1.1286007165908813,
                           -1.099538658544285]

        # values on different arch
        expected_variance = [0.45748308300971985,
                             0.5163105076990828]

        assert(np.any(np.isclose(e.data.item(), np.array(expected_energy))))
        assert(np.any(np.isclose(v.data.item(), np.array(expected_variance))))

    def test3_wf_opt(self):
        self.solver.sampler = self.sampler

        self.solver.configure(track=['local_energy'],
                              loss='energy', grad='auto')
        obs = self.solver.run(5)
        if __PLOT__:
            plot_energy(obs.local_energy, e0=-
                        1.1645, show_variance=True)

    def test4_geo_opt(self):

        self.solver.wf.ao.atom_coords[0, 2].data = torch.tensor(-0.37)
        self.solver.wf.ao.atom_coords[1, 2].data = torch.tensor(0.37)

        self.solver.configure(track=['local_energy'],
                              loss='energy', grad='auto')
        self.solver.geo_opt(5, nepoch_wf_init=10, nepoch_wf_update=5,
                            hdf5_group='geo_opt_correlated')

        # load the best model
        self.solver.wf.load(self.solver.hdf5file,
                            'geo_opt_correlated')
        self.solver.wf.eval()

        # sample and compute variables
        obs = self.solver.single_point()
        e, v = obs.energy, obs.variance

        e = e.data.numpy()
        v = v.data.numpy()

        # it might be too much to assert with the ground state energy
        assert(e > 2 * self.ground_state_energy and e < 0.)
        assert(v > 0 and v < 2.)


if __name__ == "__main__":
    # unittest.main()
    t = TestH2Correlated()
    t.setUp()
    t.test_0_wavefunction()
    # t.test1_single_point()
    # t.test2_single_point_hmc()
    # # t.test3_wf_opt()
    # t.test5_sampling_traj()
