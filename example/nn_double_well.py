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
import matplotlib.pyplot as plt

import numpy as np


class RBF_DW(NEURAL_WF_BASE):

    def __init__(self,nelec=1,ndim=1,ncenter=51):
        super(RBF_DW,self).__init__(nelec,ndim)

        self.ncenter = ncenter
        self.centers = torch.linspace(-5,5,self.ncenter)
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
        return (0.25*pos**4 - pos**2 + 1.).view(-1,1)

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        return 0

# wavefunction
wf = RBF_DW(ndim=1,nelec=1,ncenter=11)

#sampler
sampler = METROPOLIS(nwalkers=250, nstep=1000, 
                     step_size = 3., nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-5,'max':5})

# optimizer
opt = optim.Adam(wf.parameters(),lr=0.005,weight_decay=0.0)

# network
net = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)
pos = None
obs_dict = None

plt.ion()
fig = plt.figure()

for i in range(1):

    net.wf.fc.weight.requires_grad = True
    net.wf.rbf.centers.requires_grad = False

    pos,obs_dict = net.train(1000,
             batchsize=250,
             pos = pos,
             obs_dict = obs_dict,
             resample=100,
             ntherm=-1,
             loss = 'variance',
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

net.plot_results(obs_dict)






