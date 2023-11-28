import unittest

import numpy as np
import torch
import torch.optim as optim


from qmctorch.sampler import Metropolis
from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel
from qmctorch.solver import Solver

__PLOT__ = True


class TestH2GeoOpt(unittest.TestCase):

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

        # jastrow
        jastrow = JastrowFactor(self.mol, PadeJastrowKernel)

        # wave function
        self.wf = SlaterJastrow(self.mol,
                                kinetic='auto',
                                configs='single(2,2)',
                                jastrow=jastrow)

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

    def test_geo_opt(self):

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
        gse = -1.16
        assert(e > 2 * gse and e < 0.)
        assert(v > 0 and v < 2.)


if __name__ == "__main__":
    unittest.main()
    # t = TestH2()
    # t.setUp()
    # # t.test2_single_point_hmc()
    # # t.test1_single_point()
    # t.test3_wf_opt()
    # # t.test5_sampling_traj()
