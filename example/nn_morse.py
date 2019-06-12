import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import torch.optim as optim

from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF_BASE
from pyCHAMP.wavefunction.rbf import RBF
from pyCHAMP.solver.deepqmc import DeepQMC
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS
from pyCHAMP.solver.mesh import adaptive_mesh_1d as mesh

import matplotlib.pyplot as plt

import numpy as np

class RBF_MORSE(NEURAL_WF_BASE):

    def __init__(self,nelec=1,ndim=1,ncenter=51):
        super(RBF_MORSE,self).__init__(nelec,ndim)

        
        #self.centers = torch.linspace(-2,6,self.ncenter)
        self.centers = torch.tensor(mesh(self.nuclear_potential,-2,5,ncenter,loss='curvature'))
        self.ncenter = len(self.centers)
        self.rbf = RBF(self.ndim_tot, self.ncenter,centers=self.centers,opt_centers=False)
        #self.fc = weight_norm(nn.Linear(self.ncenter, 1, bias=False),'weight')
        self.fc = nn.Linear(self.ncenter, 1, bias=False)

        #self.fc.weight.data.fill_(1.)
        nn.init.uniform_(self.fc.weight,0,1)
        #nn.init.xavier_uniform_(self.fc.weight)

    def forward(self,x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''

        batch_size = x.shape[0]
        x = x.view(batch_size,-1,self.ndim)
        x = self.rbf(x)
        x = self.fc(x)
        return x.view(-1,1)

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi
        '''
        
        return 0.5*(torch.exp(-2.*(pos)) - 2.*torch.exp(-pos)).view(-1,1)

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        return 0

def morse_sol(pos):
    '''Analytical solution of the morse oscillator.'''
    vn = np.exp(-np.exp(-pos)-0.5*pos)
    return vn/np.linalg.norm(vn)

# wavefunction
wf = RBF_MORSE(ndim=1,nelec=1,ncenter=11)

#sampler
sampler = METROPOLIS(nwalkers=250, nstep=1000, 
                     step_size = 3., nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-2,'max':5})

# optimizer
opt = optim.Adam(wf.parameters(),lr=0.01,weight_decay=0.0)

# network
net = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)
pos = None
obs_dict = None

plt.ion()
fig = plt.figure()

for i in range(1):

    net.wf.fc.weight.requires_grad = True
    net.wf.rbf.centers.requires_grad = False

    pos,obs_dict = net.train(250,
             batchsize=250,
             pos = pos,
             obs_dict = obs_dict,
             resample=100,
             ntherm=-1,
             loss = 'variance',
             sol=morse_sol,
             fig=fig)

    # net.wf.fc.weight.requires_grad = False
    # net.wf.rbf.centers.requires_grad = True

    # pos,obs_dict = net.train(50,
    #          batchsize=250,
    #          pos = pos,
    #          obs_dict = obs_dict,
    #          resample=100,
    #          ntherm=-1,
    #          loss = 'variance',
    #          sol=morse_sol,
    #          fig=fig)

net.plot_results(obs_dict,morse_sol,e0=-0.125)






