import unittest

import numpy as np
import torch
import torch.optim as optim
import horovod.torch as hvd
from mpi4py import MPI

from qmctorch.sampler import Metropolis
from qmctorch.solver import SolverMPI
from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow
from qmctorch.utils import set_torch_double_precision


class TestH2Hvd(unittest.TestCase):

    def setUp(self):
        hvd.init()

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
            basis='sto-3g',
            rank=hvd.local_rank())

        # wave function
        self.wf = SlaterJastrow(self.mol, kinetic='jacobi',
                                configs='cas(2,2)',
                                cuda=False)

        # sampler
        self.sampler = Metropolis(
            nwalkers=200,
            nstep=200,
            step_size=0.2,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain('atomic'),
            move={
                'type': 'all-elec',
                'proba': 'normal'})

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        # solver
        self.solver = SolverMPI(wf=self.wf, sampler=self.sampler,
                                                 optimizer=self.opt, rank=hvd.rank())

        # ground state energy
        self.ground_state_energy = -1.16

        # ground state pos
        self.ground_state_pos = 0.69

    def test_single_point(self):
        self.solver.wf.ao.atom_coords[0, 2] = -self.ground_state_pos
        self.solver.wf.ao.atom_coords[1, 2] = self.ground_state_pos

        # sample and compute observables
        obs = self.solver.single_point()
        e, v = obs.energy, obs.variance

        e = e.data.item()
        v = v.data.item()

        assert np.isclose(e, -1.15, 0.2)
        assert 0 < v < 2

    def test_wf_opt(self):
        self.solver.wf.ao.atom_coords[0, 2] = -self.ground_state_pos
        self.solver.wf.ao.atom_coords[1, 2] = self.ground_state_pos

        self.solver.configure(track=['local_energy'], freeze=['ao', 'mo'],
                              loss='energy', grad='auto',
                              ortho_mo=False, clip_loss=False,
                              resampling={'mode': 'update',
                                          'resample_every': 1,
                                          'nstep_update': 50})
        self.solver.run(10)

        MPI.COMM_WORLD.barrier()

        self.solver.wf.load(self.solver.hdf5file, 'wf_opt')
        self.solver.wf.eval()

        obs = self.solver.single_point()

        e, v = obs.energy, obs.variance

        e = e.data.numpy()
        v = v.data.numpy()

        assert np.isclose(e, -1.15, 0.2)
        assert 0 < v < 2

    # def test_geo_opt(self):
    #     self.solver.wf.ao.atom_coords[0, 2].data = torch.as_tensor(-0.37)
    #     self.solver.wf.ao.atom_coords[1, 2].data = torch.as_tensor(0.37)
    #
    #     self.solver.configure(track=['local_energy'], freeze=['ao', 'mo'],
    #                           loss='energy', grad='auto',
    #                           ortho_mo=False, clip_loss=False,
    #                           resampling={'mode': 'update',
    #                                       'resample_every': 1,
    #                                       'nstep_update': 50})
    #     self.solver.geo_opt(5, nepoch_wf_init=10, nepoch_wf_update=5)
    #
    #     MPI.COMM_WORLD.barrier()
    #
    #     # load the best model
    #     self.solver.wf.load(self.solver.hdf5file, 'geo_opt')
    #     self.solver.wf.eval()
    #
    #     # sample and compute variables
    #     obs = self.solver.single_point()
    #     e, v = obs.energy, obs.variance
    #
    #     e = e.data.numpy()
    #     v = v.data.numpy()
    #
    #     assert np.isclose(e, -1.15, 0.2)
    #     assert (0 < v < 2.)


if __name__ == "__main__":
    t = TestH2Hvd()
    t.setUp()
    t.test_single_point()
    t.test_wf_opt()
    # t.test_geo_opt()
