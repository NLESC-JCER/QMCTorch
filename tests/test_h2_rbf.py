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
from pyCHAMP.wavefunction.wave_modules import SlaterPooling, TwoBodyJastrowFactor, ElectronDistance

import unittest

class RBF_H2(NEURAL_WF_BASE):

    def __init__(self,centers,sigma):
        super(RBF_H2,self).__init__(2,3)

        # basis function
        self.centers = centers
        self.ncenters = len(centers)

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, 
                       self.ncenters, 
                       centers=centers, 
                       sigma=sigma,
                       nelec=self.nelec)
        
        # define the mo layer
        self.mo = nn.Linear(self.ncenters, self.ncenters, bias=False)
        mo_coeff =  torch.sqrt(torch.tensor([1./2.]))  * torch.tensor([[1.,1.],[1.,-1.]])
        self.mo.weight = nn.Parameter(mo_coeff.transpose(0,1))

        # jastrow
        self.edist = ElectronDistance(2,3)
        self.jastrow = TwoBodyJastrowFactor(1,1)

        # define the SD pooling layer
        self.configs = (torch.LongTensor([np.array([0])]), torch.LongTensor([np.array([0])]))
        self.nci = 1
        self.pool = SlaterPooling(self.configs,1,1)

        # define the linear layer
        self.fc = nn.Linear(self.nci, 1, bias=False)
        self.fc.weight.data.fill_(1.)
        self.fc.clip = False
        

    def forward(self,x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''
        
        edist  = self.edist(x)
        J = self.jastrow(edist)

        x = self.rbf(x)
        x = self.mo(x)
        #x = self.pool(x) #<- issue with batch determinant
        x = (x[:,0,0]*x[:,1,0]).view(-1,1)
        return J*x


    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi

        TODO : vecorize that !! The solution below doesn't really wirk :(def plot_observable(obs_dict,e0=None,ax=None):)
        '''
        
        p = torch.zeros(pos.shape[0])
        for ielec in range(self.nelec):
            pelec = pos[:,(ielec*self.ndim):(ielec+1)*self.ndim]
            for iatom in range(len(self.centers)):
                patom = self.rbf.centers[iatom,:]

                r = torch.sqrt(   ((pelec-patom)**2).sum(1)  ) + 1E-6
                p += (-1./r)

        return p.view(-1,1)
   

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''

        pot = torch.zeros(pos.shape[0])
        
        for ielec1 in range(self.nelec-1):
            epos1 = pos[:,ielec1*self.ndim:(ielec1+1)*self.ndim]
            
            for ielec2 in range(ielec1+1,self.nelec):
                epos2 = pos[:,ielec2*self.ndim:(ielec2+1)*self.ndim]
                
                r = torch.sqrt( ((epos1-epos2)**2).sum(1) ) + 1E-6
                pot = (1./r) 

        return pot.view(-1,1)

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

class TestH2(unittest.TestCase):

    def setUp(self):

        # optimal parameters
        self.opt_r = 0.69 # the two h are at +0.69 and -0.69
        self.opt_sigma = 1.24

        # wavefunction
        centers = torch.tensor([[0.,0.,-self.opt_r],[0.,0.,self.opt_r]])
        sigma = torch.tensor([self.opt_sigma,self.opt_sigma])

        self.wf = RBF_H2(centers = centers,
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
        self.opt = optim.Adam(self.wf.parameters(),lr=0.005)

        # network
        self.net = DeepQMC(wf=self.wf,sampler=self.mh_sampler,optimizer=self.opt)


        # ground state energy
        self.ground_state_energy = -1.16

    def test_single_point_metropolis_hasting_sampling(self):

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
        s = 1.24
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

        assert(emin<-1.1)


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
        pos,obs_dict = self.net.train(150,
                 batchsize=1000,
                 pos = None,
                 obs_dict = obs_dict,
                 resample = 250,
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
        assert (e < -1.1)
        assert (v < 0.1)

if __name__ == "__main__":
    unittest.main()









