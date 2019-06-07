import torch
from torch import nn
from torch.autograd import Variable
from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF_BASE
from pyCHAMP.wavefunction.rbf import RBF
from pyCHAMP.solver.deepqmc import DeepQMC
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS
import matplotlib.pyplot as plt

import numpy as np


class RBF_HO1D(NEURAL_WF_BASE):

    def __init__(self,nelec=1,ndim=1,ncenter=51):
        super(RBF_HO1D,self).__init__(nelec,ndim)

        self.ncenter = ncenter
        self.centers = torch.linspace(-5,5,self.ncenter)
        self.rbf = RBF(self.ndim_tot, self.ncenter,centers=self.centers)
        self.fc = nn.Linear(self.ncenter, 1, bias=False)

        nn.init.uniform_(self.fc.weight,0,1)

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
        return x

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi
        '''
        return (0.5*pos**2).flatten()

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        return 0

def ho1d_sol(pos):
    return np.exp(-0.5*pos**2)

wf = RBF_HO1D(ndim=1,nelec=1)

sampler = METROPOLIS(nwalkers=250, nstep=1000, 
                     step_size = 3., nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-5,'max':5})

net = DeepQMC(wf=wf,sampler=sampler)
net.train(250,ntherm=-1,loss = 'variance',sol=ho1d_sol)




