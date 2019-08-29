import autograd.numpy as np 
from autograd import elementwise_grad as egrad
from autograd import hessian, jacobian
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

from time import time

class WaveFunction(nn.Module):

    def __init__(self,nelec, ndim, kinetic='auto'):

        super(WaveFunction, self).__init__()

        self.ndim = ndim
        self.nelec = nelec
        self.ndim_tot = self.nelec*self.ndim
        self.kinetic = kinetic

    def forward(self,x):
        
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''

        raise NotImplementedError()  

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

        Returns: values of Ven * psi
        '''
        raise NotImplementedError()

    def nuclear_repulsion(self):
        '''Compute the nuclear repulsion term 

        Returns: values of Vnn * psi
        '''
        raise NotImplementedError()        


    def kinetic_energy(self,pos,out=None):
        '''Main switch for the kinetic energy.'''

        if self.kinetic == 'auto':
            return self.kinetic_energy_autograd(pos,out)
        elif self.kinetic == 'fd':
            return self.kinetic_energy_finite_difference(pos)
        else:
            raise ValueError('kinetif %s not recognized' %self.kinetic)

    def kinetic_energy_autograd(self,pos,out=None):
        '''Compute the second derivative of the network
        output w.r.t the value of the input. 

        This is to compute the value of the kinetic operator.

        Args:
            pos: position of the electron
            out : preomputed values of the wf at pos

        Returns:
            values of nabla^2 * Psi
        '''

        if out is None:
            out = self.forward(pos)

        # compute the jacobian            
        z = Variable(torch.ones(out.shape))
        
        jacob = grad(out,pos,grad_outputs=z,create_graph=True,only_inputs=False)[0]
        
        # compute the diagonal element of the Hessian
        z = Variable(torch.ones(jacob.shape[0]))
        hess = torch.zeros(jacob.shape[0])
        
        for idim in range(jacob.shape[1]):
            tmp = grad(jacob[:,idim],pos,
                      grad_outputs=z,
                      create_graph=True,
                      allow_unused=True,
                      only_inputs=False)[0]    
            hess += tmp[:,idim]
        
        return -0.5 * hess.view(-1,1)
    
    def kinetic_energy_finite_difference(self,pos,eps=1E-3):
        '''Compute the second derivative of the network
        output w.r.t the value of the input using finite difference.

        This is to compute the value of the kinetic operator.

        Args:
            pos: position of the electron
            out : preomputed values of the wf at pos
            eps : psilon for numerical derivative
        Returns:
            values of nabla^2 * Psi
        '''
        
        nwalk = pos.shape[0]
        ndim = pos.shape[1]
        out = torch.zeros(nwalk,1)

        for icol in range(ndim):

            pos_tmp = pos.clone()
            feps = -2*self.forward(pos_tmp)

            pos_tmp = pos.clone()
            pos_tmp[:,icol] += eps
            feps += self.forward(pos_tmp)

            pos_tmp = pos.clone()
            pos_tmp[:,icol] -= eps
            feps += self.forward(pos_tmp)

            out += feps/(eps**2)

        return -0.5*out.view(-1,1)


    def local_energy_save(self,pos):
        ''' local energy of the sampling points.'''
        return self.kinetic_energy(pos)/self.forward(pos) \
             + self.nuclear_potential(pos)  \
             + self.electronic_potential(pos) \
             + self.nuclear_repulsion()

    def local_energy(self,pos):
        ''' local energy of the sampling points.'''
        #t0 = time()    
        wf = self.forward(pos)
        ke = self.kinetic_energy(pos,out=wf)
        #print('Kinetic done in %f' %(time()-t0))
        
        return ke/wf \
             + self.nuclear_potential(pos)  \
             + self.electronic_potential(pos) \
             + self.nuclear_repulsion()        

    def energy(self,pos):
        '''Total energy for the sampling points.'''
        return torch.mean(self.local_energy(pos)) 

    def variance(self, pos):
        '''Variance of the energy at the sampling points.'''
        return torch.var(self.local_energy(pos))

    def pdf(self,pos):
        '''density of the wave function.'''
        return (self.forward(pos)**2).reshape(-1)
