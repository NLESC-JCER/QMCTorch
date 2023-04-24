from tests.wavefunction.test_slaterjastrow import TestSlaterJastrow
import unittest

import numpy as np
import torch
import torch.optim as optim

from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver
from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrowBackFlow
from qmctorch.utils import set_torch_double_precision


class TestLiHBackFlowPySCF(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)
        np.random.seed(0)
        set_torch_double_precision()

        # molecule
        self.mol = Molecule(
            atom='Li 0 0 0; H 0 0 3.015',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g')

        # wave function
        self.wf = SlaterJastrowBackFlow(self.mol, kinetic='jacobi',
                                        configs='single_double(2,2)',
                                        orbital_dependent_backflow=True,
                                        include_all_mo=True)

        # fc weights
        self.wf.fc.weight.data = torch.rand(self.wf.fc.weight.shape)

        # jastrow weights
        self.wf.jastrow.jastrow_kernel.weight.data = torch.rand(
            self.wf.jastrow.jastrow_kernel.weight.shape)

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

        # artificial pos
        self.nbatch = 10
        self.pos = torch.as_tensor(np.random.rand(
            self.nbatch, self.wf.nelec*3))
        self.pos.requires_grad = True

    def test_0_wavefunction(self):

        eauto = self.wf.kinetic_energy_autograd(self.pos)
        ejac = self.wf.kinetic_energy_jacobi(self.pos)
        print(torch.stack([eauto, ejac], axis=1).squeeze())
        assert torch.allclose(
            eauto.data, ejac.data, rtol=1E-4, atol=1E-4)

    def test1_single_point(self):

        # sample and compute observables
        obs = self.solver.single_point()
        e, v = obs.energy, obs.variance

    def test2_wf_opt_grad_auto(self):
        self.solver.sampler = self.sampler

        self.solver.configure(track=['local_energy'],
                              loss='energy', grad='auto')
        obs = self.solver.run(5)

    def test3_wf_opt_grad_manual(self):
        self.solver.sampler = self.sampler

        self.solver.configure(track=['local_energy', 'parameters'],
                              loss='energy', grad='manual')
        obs = self.solver.run(5)


if __name__ == "__main__":
    # unittest.main()
    t = TestLiHBackFlowPySCF()
    t.setUp()
    t.test3_wf_opt_grad_manual()
