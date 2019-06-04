import autograd.numpy as np 
from autograd import elementwise_grad as egrad
from autograd import hessian, jacobian
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

class NEURAL_WF(nn.Module):

    def __init__(self,nelec, ndim):

        super(NEURAL_WF, self).__init__()

        self.ndim = ndim
        self.nelec = nelec
        self.ndim_tot = self.nelec*self.ndim
        
        self.eps = 1E-6
        
        self.fc1 = nn.Linear(self.ndim_tot, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.fc1.bias.data.fill_(0.00)
        self.fc2.bias.data.fill_(0.00)
        self.fc3.bias.data.fill_(0.00)

    def forward(self,x):
        
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''

        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = (self.fc1(x))
        x = (self.fc2(x))
        x = torch.exp(-self.fc3(x)**2)
        return x

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
        return (0.5*pos**2).flatten()
        #raise NotImplementedError()


    def kinetic_autograd(self,pos):

        out = self.forward(pos)
        z = Variable(torch.ones(out.shape))
        jacob = grad(out,pos,grad_outputs=z,create_graph=True)[0]
        hess = grad(jacob.sum(),pos,create_graph=True)[0]
        return hess.sum(1)


    def applyK(self,pos):
        '''Comute the result of H * psi

        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * pis
        ''' 
        Kpsi = -0.5*self.kinetic_autograd(pos) 
        return Kpsi

    
    def local_energy(self,pos):
        ''' local energy of the sampling points.'''
        return self.applyK(pos)/self.forward(pos) \
             + self.nuclear_potential(pos)  
               # + self.electronic_potential(pos)

    def energy(self,pos):
        '''Total energy for the sampling points.'''
        return torch.mean(self.local_energy(pos))

    def variance(self, pos):
        '''Variance of the energy at the sampling points.'''
        return torch.var(self.local_energy(pos))

    def pdf(self,pos):
        '''density of the wave function.'''
        return (self.forward(pos)**2).reshape(-1)



class WaveNet1D(nn.Module):

    def __init__(self):

        super(WaveNet1D, self).__init__()
        self.fc1 = nn.Linear(1, 64)
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
