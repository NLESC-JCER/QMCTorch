import torch
import torch.optim as optim

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.solver import SolverOrbital
from qmctorch.sampler import Metropolis, Hamiltonian

import numpy as np
import unittest


class TestH2ADF(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)

        # molecule
        self.mol = Molecule(
            atom='H 0 0 -0.69; H 0 0 0.69',
            unit='bohr',
            calculator='adf',
            basis='dzp')

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

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        # solver
        self.solver = SolverOrbital(wf=self.wf, sampler=self.sampler,
                                    optimizer=self.opt)

        # ground state energy
        self.ground_state_energy = -1.16

        # value of the energy for seed=0
        self.expected_energy = -1.1572532653808594
        self.expected_variance = 0.05085879936814308

        # ground state pos
        self.ground_state_pos = 0.69

    def test_single_point(self):

        self.solver.wf.ao.atom_coords[0, 2] = -self.ground_state_pos
        self.solver.wf.ao.atom_coords[1, 2] = self.ground_state_pos
        self.solver.sampler = self.sampler

        # sample and compute observables
        _, e, v = self.solver.single_point()

        print('Energy   :', e.data)
        print('Variance :', v.data)

        assert(e > 2 * self.ground_state_energy and e < 0.)
        assert(v > 0 and v < 5.)

        assert(np.isclose(e.data, self.expected_energy))
        assert(np.isclose(v.data.item(), self.expected_variance))


if __name__ == "__main__":
    # unittest.main()
    t = TestH2ADF()
    t.setUp()
    t.test_single_point()
