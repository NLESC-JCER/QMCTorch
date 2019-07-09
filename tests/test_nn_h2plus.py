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

class RBF_H2p(NEURAL_WF_BASE):

    def __init__(self,centers,sigma):
        super(RBF_H2p,self).__init__(1,3)

        # get the RBF centers 
        self.ncenter = len(centers)

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, 
                       self.ncenter, 
                       centers=centers,
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
        c0 = self.rbf.centers[0,:]
        c1 = self.rbf.centers[1,:]

        r0 = torch.sqrt(   ((pos-c0)**2).sum(1)  ) + 1E-3
        r1 = torch.sqrt(   ((pos-c1)**2).sum(1)  ) + 1E-3

        p0 = (-1./r0).view(-1,1)
        p1 = (-1./r1).view(-1,1)
        
        return p0 + p1
   

    def electronic_potential(self,pos):
        return 0

    def nuclear_repulsion(self):
        c0 = self.rbf.centers[0,:]
        c1 = self.rbf.centers[1,:]
        rnn = torch.sqrt(   ((c0-c1)**2).sum()  )
        return (1./rnn).view(-1,1)

    def atomic_distance(self,pos=None):
        c0 = self.rbf.centers[0,:]
        c1 = self.rbf.centers[1,:]
        return torch.sqrt(   ((c0-c1)**2).sum()  )

    def get_sigma(self,pos=None):
        return self.rbf.sigma.data[0]

class TestH2plus(unittest.TestCase):

    def setUp(self):

        # optimal parameters
        self.opt_r = 0.97 # the two h are at +0.97 and -0.97
        self.opt_sigma = 1.20

        # wavefunction
        centers = torch.tensor([[0.,0.,-self.opt_r],[0.,0.,self.opt_r]])
        sigma = torch.tensor([self.opt_sigma,self.opt_sigma])

        self.wf = RBF_H2p(centers = centers,
                          sigma   = sigma )

        #sampler
        self.mh_sampler = METROPOLIS(nwalkers=1000, nstep=1000, 
                             step_size = 0.5, nelec = self.wf.nelec, 
                             ndim = self.wf.ndim, domain = {'min':-5,'max':5})

        #sampler
        self.hmc_sampler = HAMILTONIAN(nwalkers=1000, nstep=200, 
                             step_size = 0.1, nelec = self.wf.nelec, 
                             ndim = self.wf.ndim, domain = {'min':-5,'max':5}, L=5)

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(),lr=0.001)

        # network
        self.net = DeepQMC(wf=self.wf,sampler=self.mh_sampler,optimizer=self.opt)


        # ground state energy
        self.ground_state_energy = -0.597

    def test_single_point_metropolis_hasting_sampling(self):

        # sample and compute observables
        pos = self.net.sample(ntherm=-1,with_tqdm=False)
        e = self.wf.energy(pos).detach().numpy()
        v = self.wf.variance(pos).detach().numpy()

        print('Energy   :', e)
        print('Variance :', v)

        assert np.allclose([e,v],[self.ground_state_energy,0],atol=1E-1)

    def test_single_point_hamiltonian_mc_sampling(self):

        #switch to HMC sampling
        self.net.sampler = self.hmc_sampler

        # sample and compute observables
        pos = self.net.sample(ntherm=-1,with_tqdm=False)
        e = self.wf.energy(pos).detach().numpy()
        v = self.wf.variance(pos).detach().numpy()

        print('Energy   :', e)
        print('Variance :', v)

        assert np.allclose([e,v],[self.ground_state_energy,0],atol=1E-1)

    def test_energy_curve(self):

        #switch to MH sampling
        self.net.sampler = self.mh_sampler

        # fix the sigma of the AO
        s = 1.20
        self.net.wf.rbf.sigma.data[:] = s 

        X = np.linspace(0.1,2,25)
        emin = 1E3
        for x in X:

            # move the atoms
            self.net.wf.rbf.centers.data[0,2] = -x
            self.net.wf.rbf.centers.data[1,2] = x

            pos = Variable(self.net.sample())
            pos.requires_grad = True
            e = self.net.wf.energy(pos)

            if e<emin:
                emin = e            

        assert(emin<-0.55)


    def test_sigma_optimization(self):

            #switch to MH sampling
            self.net.sampler = self.mh_sampler

            # move the atoms
            x = 0.97
            self.net.wf.rbf.centers.data[0,2] = -x
            self.net.wf.rbf.centers.data[1,2] = x

            # fix the sigma of the AO
            s = 1.
            self.net.wf.rbf.sigma.data[:] = s 

            # do not optimize the weight of the FC layer
            # optimize the pos of the centers
            # do not optimize the sigma of the AO
            self.net.wf.fc.weight.requires_grad = False
            self.net.wf.rbf.centers.requires_grad = False
            self.net.wf.rbf.sigma.requires_grad = True

            # define the observable we want
            obs_dict = {'local_energy':[],
                        'atomic_distance':[],
                        'get_sigma':[]}
            # train
            pos,obs_dict = self.net.train(100,
                     batchsize=1000,
                     pos = None,
                     obs_dict = obs_dict,
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

            # it might be too much to assert with the ground state energy
            assert (e < -0.5)
            assert (v < 0.1)


    def test_geo_optimization(self):

        #switch to MH sampling
        self.net.sampler = self.mh_sampler

        # move the atoms
        x = 0.5
        self.net.wf.rbf.centers.data[0,2] = -x
        self.net.wf.rbf.centers.data[1,2] = x

        # fix the sigma of the AO
        s = 1.20
        self.net.wf.rbf.sigma.data[:] = s 

        # do not optimize the weight of the FC layer
        # optimize the pos of the centers
        # do not optimize the sigma of the AO
        self.net.wf.fc.weight.requires_grad = False
        self.net.wf.rbf.centers.requires_grad = True
        self.net.wf.rbf.sigma.requires_grad = False

        # define the observable we want
        obs_dict = {'local_energy':[],
                    'atomic_distance':[],
                    'get_sigma':[]}
        # train
        pos,obs_dict = self.net.train(200,
                 batchsize=1000,
                 pos = None,
                 obs_dict = obs_dict,
                 resample = 100,
                 resample_from_last = True,
                 resample_every = 1,
                 ntherm = -1,
                 loss = 'energy',
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

        # it might be too much to assert with the ground state energy
        assert (e < -0.5)
        assert (v < 0.1)

if __name__ == "__main__":
    unittest.main()









