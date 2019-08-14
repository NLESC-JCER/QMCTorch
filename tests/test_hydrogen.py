import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF_BASE
from pyCHAMP.wavefunction.rbf import RBF_Slater_NELEC as RBF
from pyCHAMP.solver.deepqmc import DeepQMC
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS
from pyCHAMP.sampler.hamiltonian import HAMILTONIAN_TORCH as HAMILTONIAN

import unittest

class RBF_H(NEURAL_WF_BASE):

    def __init__(self,centers,sigma):
        super(RBF_H,self).__init__(1,3)

        # get the RBF centers 
        self.centers = centers
        self.ncenter = len(self.centers)

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, 
                       self.ncenter, 
                       centers=self.centers,
                       nelec=self.nelec,
                       sigma=sigma)
        
        # define the fc layer
        self.fc = nn.Linear(self.ncenter, 1, bias=False)

        # initiaize the fc layer
        self.fc.weight.data.fill_(2.)
        self.fc.clip = False
        

    def forward(self,x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons
        Args:
            parameters : variational param of the wf
            pos: position of the electrons
        Returns: values of psi
        '''

        x = self.rbf(x)
        x = self.fc(x)
        return x.view(-1,1)

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron
        Returns: values of V * psi
        '''
        c0 = self.centers[0,:]
        r0 = torch.sqrt(   ((pos-c0)**2).sum(1)  ) 
        p0 = (-1./r0).view(-1,1)
        return p0 
   

    def electronic_potential(self,pos):
        return 0

    def nuclear_repulsion(self):
        return 0

class TestHydrogen(unittest.TestCase):

    def setUp(self):

        # wavefunction
        self.wf = RBF_H(centers=torch.tensor([[0.,0.,0.]]),
                        sigma=torch.tensor([1.]) )

        #sampler
        self.mh_sampler = METROPOLIS(nwalkers=1000, nstep=1000, 
                             step_size = 3., nelec = self.wf.nelec, 
                             ndim = self.wf.ndim, domain = {'min':-5,'max':5})

        #sampler
        self.hmc_sampler = HAMILTONIAN(nwalkers=1000, nstep=1000, 
                             step_size = 0.01, nelec = self.wf.nelec, 
                             ndim = self.wf.ndim, domain = {'min':-5,'max':5}, L=5)

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(),lr=0.01)

        # network
        self.net = DeepQMC(wf=self.wf,sampler=self.mh_sampler,optimizer=self.opt)


        # ground state energy
        self.ground_state_energy = -0.5

    def test_single_point_metropolis_hasting_sampling(self):

        # sample and compute observables
        pos = self.net.sample(ntherm=-1,with_tqdm=False)
        e = self.wf.energy(pos).detach().numpy()
        v = self.wf.variance(pos).detach().numpy()
        assert np.allclose([e,v],[self.ground_state_energy,0],atol=1E-6)

    def test_single_point_hamiltonian_mc_sampling(self):

        #switch to HMC sampling
        self.net.sampler = self.hmc_sampler

        # sample and compute observables
        pos = self.net.sample(ntherm=-1,with_tqdm=False)
        e = self.wf.energy(pos).detach().numpy()
        v = self.wf.variance(pos).detach().numpy()
        assert np.allclose([e,v],[self.ground_state_energy,0],atol=1E-6)

    def test_optimization(self):

        #switch to MH sampling
        self.net.sampler = self.mh_sampler

        # optimize the weight of the FC layer
        # do not optimize the pos of the centers
        self.net.wf.fc.weight.requires_grad = False
        self.net.wf.rbf.sigma.requires_grad = True

        # modify the sigma 
        self.net.wf.rbf.sigma.data[0] = 1.5
        
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
        assert np.allclose([e,v],[self.ground_state_energy,0],atol=1E-3)

if __name__ == "__main__":
    unittest.main()









