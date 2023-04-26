import unittest

import numpy as np
import torch
import torch.optim as optim

from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver
from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow
from ..path_utils import PATH_TEST


class TestH2ADFJacobi(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)

        # molecule
        path_hdf5 = (
            PATH_TEST / 'hdf5/H2_adf_dzp.hdf5').absolute().as_posix()
        self.mol = Molecule(load=path_hdf5)

        # wave function
        self.wf = SlaterJastrow(self.mol, kinetic='jacobi',
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

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        # solver
        self.solver = Solver(wf=self.wf, sampler=self.sampler,
                                          optimizer=self.opt)

        # ground state energy
        self.ground_state_energy = -1.16

        # ground state pos
        self.ground_state_pos = 0.69

    def test_single_point(self):

        self.solver.wf.ao.atom_coords[0, 2] = -self.ground_state_pos
        self.solver.wf.ao.atom_coords[1, 2] = self.ground_state_pos
        self.solver.sampler = self.sampler

        # sample and compute observables
        obs = self.solver.single_point()
        e, v = obs.energy, obs.variance
        print(e.data.item(), v.data.item())

        # vals on different archs
        expected_energy = [-1.1571345329284668,
                           -1.1501641653648578]

        expected_variance = [0.05087674409151077,
                             0.05094174843043177]

        assert(np.any(np.isclose(e.data.item(), np.array(expected_energy))))
        assert(np.any(np.isclose(v.data.item(), np.array(expected_variance))))

    def test_wf_opt_auto_grad(self):

        self.solver.configure(track=['local_energy'],
                              loss='energy', grad='auto')
        obs = self.solver.run(5)

    def test_wf_opt_manual_grad(self):

        self.solver.configure(track=['local_energy'],
                              loss='energy', grad='manual')
        obs = self.solver.run(5)


if __name__ == "__main__":
    unittest.main()
    # t = TestH2ADFJacobi()
    # t.setUp()
    # t.test_single_point()
