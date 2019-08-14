import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF_BASE
from pyCHAMP.wavefunction.rbf import RBF
from pyCHAMP.solver.deepqmc import DeepQMC
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS
from pyCHAMP.sampler.hamiltonian import HAMILTONIAN_TORCH as HAMILTONIAN

import unittest

class RBF_HO1D(NEURAL_WF_BASE):

    def __init__(self,nelec=1,ndim=1,ncenter=51):
        super(RBF_HO1D,self).__init__(nelec,ndim)

        # get the RBF centers 
        self.centers = torch.linspace(-5,5,ncenter).view(-1,1)
        self.ncenter = len(self.centers)

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, self.ncenter,
                      centers=self.centers, opt_centers=False,
                      sigma = 1.)
        
        # define the fc layer
        self.fc = nn.Linear(self.ncenter, 1, bias=False)
        self.fc.clip = True

        # initiaize the fc layer
        self.fc.weight.data.fill_(0.)
        self.fc.weight.data[0,2] = 1.

    def forward(self,x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''

        batch_size = x.shape[0]
        x = self.rbf(x)
        x = self.fc(x)
        return x.view(-1,1)

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi
        '''
        return (0.5*pos**2).flatten().view(-1,1)

    def electronic_potential(self,pos):
        return 0

    def nuclear_repulsion(self):
        return 0

class TestRbfNetworkHarmonicOscillator1D(unittest.TestCase):

    def setUp(self):

        # wavefunction
        self.wf = RBF_HO1D(ndim=1,nelec=1,ncenter=5)

        #sampler
        self.mh_sampler = METROPOLIS(nwalkers=250, nstep=1000, 
                             step_size = 3., nelec = self.wf.nelec, 
                             ndim = self.wf.ndim, domain = {'min':-5,'max':5})

        #sampler
        self.hmc_sampler = HAMILTONIAN(nwalkers=250, nstep=1000, 
                             step_size = 0.01, nelec = self.wf.nelec, 
                             ndim = self.wf.ndim, domain = {'min':-5,'max':5}, L=5)

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(),lr=0.01)

        # network
        self.net = DeepQMC(wf=self.wf,sampler=self.mh_sampler,optimizer=self.opt)

    def test_single_point_metropolis_hasting_sampling(self):

        # initiaize the fc layer
        self.net.wf.fc.weight.data.fill_(0.)
        self.net.wf.fc.weight.data[0,2] = 1.

        # sample and compute observables
        pos = self.net.sample(ntherm=-1,with_tqdm=False)
        e = self.wf.energy(pos).detach().numpy()
        v = self.wf.variance(pos).detach().numpy()
        assert np.allclose([e,v],[0.5,0],atol=1E-6)

    def test_single_point_hamiltonian_mc_sampling(self):

        #switch to HMC sampling
        self.net.sampler = self.hmc_sampler

        # initiaize the fc layer
        self.net.wf.fc.weight.data.fill_(0.)
        self.net.wf.fc.weight.data[0,2] = 1.

        # sample and compute observables
        pos = self.net.sample(ntherm=-1,with_tqdm=False)
        e = self.wf.energy(pos).detach().numpy()
        v = self.wf.variance(pos).detach().numpy()
        assert np.allclose([e,v],[0.5,0],atol=1E-6)

    def test_optimization(self):

        #switch to MH sampling
        self.net.sampler = self.mh_sampler

        # optimize the weight of the FC layer
        # do not optimize the pos of the centers
        self.net.wf.fc.weight.requires_grad = True
        self.net.wf.rbf.centers.requires_grad = False

        # randomize the weights
        nn.init.uniform_(self.wf.fc.weight,0,1)

        # train
        pos,obs_dict = self.net.train(250,
                 batchsize=250,
                 pos = None,
                 obs_dict = None,
                 resample = 100,
                 resample_from_last = True,
                 resample_every = 1,
                 ntherm = -1,
                 loss = 'variance',
                 plot = None,
                 save_model = 'best_model.pth')

        # load the best model
        best_model = torch.load('best_model.pth')
        self.net.wf.load_state_dict(best_model['model_state_dict'])
        self.net.wf.eval()

        # sample and compute variables
        pos = self.net.sample()
        e = self.wf.energy(pos).detach().numpy()
        v = self.wf.variance(pos).detach().numpy()
        assert np.allclose([e,v],[0.5,0],atol=1E-3)

if __name__ == "__main__":
    unittest.main()









