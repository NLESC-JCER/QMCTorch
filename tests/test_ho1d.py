import torch
from torch import optim, nn

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.sampler.hamiltonian import Hamiltonian
from deepqmc.wavefunction.wf_potential import Potential
from deepqmc.solver.solver_potential import SolverPotential

import numpy as np
import unittest


def pot_func(pos):
    '''Potential function desired.'''
    return 0.5*pos**2


def ho1d_sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    return torch.exp(-0.5*pos**2)


class TestHarmonicOscillator1D(unittest.TestCase):

    def setUp(self):

        # wavefunction
        domain, ncenter = {'min': -5., 'max': 5.}, 11
        self.wf = Potential(pot_func, domain, ncenter,
                            fcinit='random', nelec=1, sigma=2.)

        # sampler
        self.mh_sampler = Metropolis(nwalkers=1000, nstep=2000,
                                     step_size=1., nelec=self.wf.nelec,
                                     ndim=self.wf.ndim,
                                     init={'min': -5, 'max': 5})

        # sampler
        self.hmc_sampler = Hamiltonian(nwalkers=1000, nstep=200,
                                       nelec=self.wf.nelec, ndim=self.wf.ndim,
                                       step_size=0.5,
                                       init={'min': -5, 'max': 5},
                                       L=10)

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.05)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.opt, step_size=100, gamma=0.75)

        # network
        self.solver = SolverPotential(wf=self.wf, sampler=self.mh_sampler,
                                      optimizer=self.opt,
                                      scheduler=self.scheduler)

    def test_single_point_metropolis_hasting_sampling(self):

        # initiaize the fc layer
        self.solver.wf.fc.weight.data.fill_(0.)
        self.solver.wf.fc.weight.data[0, 5] = 1.
        self.solver.wf.sigma = 2.

        # sample and compute observables
        _, e, v = self.solver.single_point()
        assert np.allclose([e.data.numpy(), v.data.numpy()], [
                           0.5, 0], atol=1E-3)

    def test_single_point_hamiltonian_mc_sampling(self):

        # switch to HMC sampling
        self.solver.sampler = self.hmc_sampler

        # initiaize the fc layer
        self.solver.wf.fc.weight.data.fill_(0.)
        self.solver.wf.fc.weight.data[0, 5] = 1.
        self.solver.wf.sigma = 2.

        # sample and compute observables
        pos, e, v = self.solver.single_point()
        assert np.allclose([e.data.numpy(), v.data.numpy()], [
                           0.5, 0], atol=1E-3)

    def test_optimization(self):

        # switch to MH sampling
        self.solver.sampler = self.mh_sampler

        # randomize the weights
        nn.init.uniform_(self.solver.wf.fc.weight, 0, 1)

        # train
        self.solver.run(50, loss='variance')

        # load the best model
        best_model = torch.load('model.pth')
        self.solver.wf.load_state_dict(best_model['model_state_dict'])
        self.solver.wf.eval()

        # sample and compute variables
        pos, e, v = self.solver.single_point()
        assert(e.data.numpy() < 2.5)
        assert(v.data.numpy() < 2.5)


if __name__ == "__main__":
    unittest.main()
