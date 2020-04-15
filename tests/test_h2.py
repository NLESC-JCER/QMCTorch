import torch
import torch.optim as optim

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.solver import SolverOrbital
from qmctorch.sampler import Metropolis, Hamiltonian

import platform

import numpy as np
import unittest


class TestH2(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)

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
        self.wf = Orbital(self.mol, kinetic='auto',
                          configs='single(2,2)',
                          use_jastrow=True)

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
        self.solver = SolverOrbital(wf=self.wf, sampler=self.sampler,
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
        _, e, v = self.solver.single_point()

        # values on different arch
        expected_energy = [-1.1464850902557373,
                           -1.14937478612449]

        # values on different arch
        expected_variance = [0.9279592633247375,
                             0.7445300449383236]

        assert(np.any(np.isclose(e.data.item(), np.array(expected_energy))))
        assert(np.any(np.isclose(v.data.item(), np.array(expected_variance))))

        # assert(e > 2 * self.ground_state_energy and e < 0.)
        # assert(v > 0 and v < 5.)

    def test2_single_point_hmc(self):

        self.solver.wf.ao.atom_coords[0, 2] = -self.ground_state_pos
        self.solver.wf.ao.atom_coords[1, 2] = self.ground_state_pos
        self.solver.sampler = self.hmc_sampler

        # sample and compute observables
        _, e, v = self.solver.single_point()

        # values on different arch
        expected_energy = [-1.077970027923584,
                           -1.027975961270174]

        # values on different arch
        expected_variance = [0.17763596773147583,
                             0.19953053065068135]

        assert(np.any(np.isclose(e.data.item(), np.array(expected_energy))))
        assert(np.any(np.isclose(v.data.item(), np.array(expected_variance))))

        # assert(e > 2 * self.ground_state_energy and e < 0.)
        # assert(v > 0 and v < 5.)

    def test3_geo_opt(self):

        self.solver.wf.ao.atom_coords[0, 2].data = torch.tensor(-0.37)
        self.solver.wf.ao.atom_coords[1, 2].data = torch.tensor(0.37)

        self.solver.configure(task='geo_opt')
        self.solver.observable(['local_energy', 'atomic_distances'])
        self.solver.run(50, loss='energy')

        # load the best model
        best_model = torch.load('model.pth')
        self.solver.wf.load_state_dict(best_model['model_state_dict'])
        self.solver.wf.eval()

        # sample and compute variables
        _, e, v = self.solver.single_point()
        e = e.data.numpy()
        v = v.data.numpy()

        # it might be too much to assert with the ground state energy
        assert(e > 2 * self.ground_state_energy and e < 0.)
        assert(v > 0 and v < 2.)


if __name__ == "__main__":
    # unittest.main()
    t = TestH2()
    t.setUp()
    t.test1_single_point()
    # t.test_single_point_hmc()
