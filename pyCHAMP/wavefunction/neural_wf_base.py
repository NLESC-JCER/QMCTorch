import autograd.numpy as np 
from autograd import elementwise_grad as egrad
from autograd import hessian, jacobian
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

class NEURAL_WF(object):

    def __init__(self,model,nelec, ndim):

        self.ndim = ndim
        self.nelec = nelec
        self.ndim_tot = self.nelec*self.ndim
        self.eps = 1E-6
        self.model = model()

    def values(self,pos):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''
        return self.model(pos)

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        raise NotImplementedError()        

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi
        '''
        raise NotImplementedError()


    def kinetic_fd(self,pos,eps=1E-6):

        '''Compute the action of the kinetic operator on the we points.
        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * psi
        '''

        nwalk = pos.shape[0]
        ndim = pos.shape[1]
        out = torch.zeros(nwalk,1)
        for icol in range(ndim):

            pos_tmp = pos.clone()
            feps = -2*self.model(pos_tmp)            

            pos_tmp = pos.clone()
            pos_tmp[:,icol] += eps
            feps += self.model(pos_tmp)
            

            pos_tmp = pos.clone()
            pos_tmp[:,icol] -= eps
            feps += self.model(pos_tmp)

            out += feps/(eps**2)

        return out

    def applyK(self,pos):
        '''Comute the result of H * psi

        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * pis
        ''' 
        Kpsi = -0.5*self.kinetic_fd(pos) 
        return Kpsi

    
    def local_energy(self,pos):
        ''' local energy of the sampling points.'''
        return self.applyK(pos)/self.values(pos) \
               + self.nuclear_potential(pos)  \
               + self.electronic_potential(pos)

    def energy(self,pos):
        '''Total energy for the sampling points.'''
        return torch.mean(self.local_energy(pos))

    def variance(self, pos):
        '''Variance of the energy at the sampling points.'''
        return torch.var(self.local_energy(pos))

    def pdf(self,pos):
        '''density of the wave function.'''
        return self.values(pos)**2



class WaveNet(nn.Module):

    def __init__(self):

        super(WaveNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.fc1.bias.data.fill_(0.001)
        self.fc2.bias.data.fill_(0.001)
        self.fc3.bias.data.fill_(0.001)

    def forward(self, x):
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
