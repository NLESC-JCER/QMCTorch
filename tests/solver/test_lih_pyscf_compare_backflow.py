from tests.wavefunction.test_slaterjastrow import TestSlaterJastrow
import unittest

import numpy as np
import torch
import torch.optim as optim

from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver
from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrowBackFlow, SlaterJastrow
from qmctorch.utils import set_torch_double_precision


def reset_generator():
    torch.manual_seed(0)
    np.random.seed(0)


class TestCompareLiHBackFlowPySCF(unittest.TestCase):

    def setUp(self):

        set_torch_double_precision()
        reset_generator()

        # molecule
        self.mol = Molecule(
            atom='Li 0 0 0; H 0 0 3.015',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g')

        # molecule
        self.mol_ref = Molecule(
            atom='Li 0 0 0; H 0 0 3.015',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g')

        # backflow wave function
        self.wf = SlaterJastrowBackFlow(self.mol,
                                        kinetic='jacobi',
                                        configs='single_double(2,2)',
                                        include_all_mo=True)
        self.wf.ao.backflow_trans.backflow_kernel.weight.data *= 0.
        self.wf.ao.backflow_trans.backflow_kernel.weight.requires_grad = False

        # normal wave function
        self.wf_ref = SlaterJastrow(self.mol_ref,
                                    kinetic='jacobi',
                                    include_all_mo=True,
                                    configs='single_double(2,2)')

        # fc weights
        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight.clone()
        self.wf_ref.fc.weight.data = self.random_fc_weight.clone()

        # jastrow weights
        self.random_jastrow_weight = torch.rand(
            self.wf.jastrow.jastrow_kernel.weight.shape)

        self.wf.jastrow.jastrow_kernel.weight.data = self.random_jastrow_weight.clone()
        self.wf_ref.jastrow.jastrow_kernel.weight.data = self.random_jastrow_weight.clone()

        reset_generator()
        # sampler
        self.sampler = Metropolis(
            nwalkers=5,
            nstep=200,
            step_size=0.05,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain('normal'),
            move={
                'type': 'all-elec',
                'proba': 'normal'})

        reset_generator()
        self.sampler_ref = Metropolis(
            nwalkers=5,
            nstep=200,
            step_size=0.05,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain('normal'),
            move={
                'type': 'all-elec',
                'proba': 'normal'})

        # optimizer
        reset_generator()
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        reset_generator()
        self.opt_ref = optim.Adam(self.wf_ref.parameters(), lr=0.01)

        # solver
        self.solver_ref = Solver(wf=self.wf_ref, sampler=self.sampler_ref,
                                              optimizer=self.opt_ref)

        self.solver = Solver(wf=self.wf, sampler=self.sampler,
                                          optimizer=self.opt)

        # artificial pos
        self.nbatch = 10
        self.pos = torch.as_tensor(np.random.rand(
            self.nbatch, self.wf.nelec*3))
        self.pos.requires_grad = True

    def test_0_wavefunction(self):

        # compute the kinetic energy using bf orb
        reset_generator()
        e_bf = self.wf.kinetic_energy_jacobi(self.pos)

        # compute the kinetic energy
        reset_generator()
        e_ref = self.wf_ref.kinetic_energy_jacobi(self.pos)

        print(torch.stack([e_bf, e_ref], axis=1).squeeze())
        assert torch.allclose(
            e_bf.data, e_ref.data, rtol=1E-4, atol=1E-4)

    def test1_single_point(self):

        # sample and compute observables
        reset_generator()
        obs = self.solver.single_point()
        e_bf, v_bf = obs.energy, obs.variance

        obs = self.solver.single_point()
        e_bf, v_bf = obs.energy, obs.variance

        # sample and compute observables
        reset_generator()
        obs_ref = self.solver_ref.single_point()
        e_ref, v_ref = obs_ref.energy, obs.variance

        obs_ref = self.solver_ref.single_point()
        e_ref, v_ref = obs_ref.energy, obs.variance

        # compare values
        assert torch.allclose(
            e_bf.data, e_ref.data, rtol=1E-4, atol=1E-4)

        assert torch.allclose(
            v_bf.data, v_ref.data, rtol=1E-4, atol=1E-4)

    def test2_wf_opt_grad_auto(self):

        nepoch = 5

        # optimize using backflow
        self.solver.configure(track=['local_energy'],
                              loss='energy', grad='auto')
        self.solver.configure_resampling(mode='never')

        reset_generator()
        obs = self.solver.run(nepoch)
        e_bf = torch.as_tensor(np.array(obs.energy))

        # optimize using ref
        self.solver_ref.configure(track=['local_energy'],
                                  loss='energy', grad='auto')
        self.solver_ref.configure_resampling(mode='never')

        reset_generator()
        obs_ref = self.solver_ref.run(nepoch)
        e_ref = torch.as_tensor(np.array(obs_ref.energy))

        assert torch.allclose(
            e_bf, e_ref, rtol=1E-4, atol=1E-4)

    def test3_wf_opt_grad_manual(self):

        nepoch = 5

        # optimize using backflow
        reset_generator()
        self.solver.configure(track=['local_energy', 'parameters'],
                              loss='energy', grad='manual')
        obs = self.solver.run(nepoch)
        e_bf = torch.as_tensor(np.array(obs.energy))

        # optimize using backflow
        reset_generator()
        self.solver_ref.configure(track=['local_energy', 'parameters'],
                                  loss='energy', grad='manual')
        obs = self.solver_ref.run(nepoch)
        e_ref = torch.as_tensor(np.array(obs.energy))

        # compare values
        assert torch.allclose(
            e_bf, e_ref, rtol=1E-4, atol=1E-4)


if __name__ == "__main__":
    # unittest.main()
    t = TestCompareLiHBackFlowPySCF()
    t.setUp()
    t.test_0_wavefunction()
    t.test1_single_point()
    t.test2_wf_opt_grad_auto()
    t.test3_wf_opt_grad_manual()
