import autograd.numpy as np 
from autograd import elementwise_grad as egrad
from functools import partial

class WF(object):

    def __init__(self,nelec, ncart):

        self.ncart = ncart
        self.nelec = nelec
        self.ndim = self.nelec*self.ncart


    def value(self,parameters,pos):
        ''' Compute the value of the wave function.

        Args:
            parameters : variational param of the wf
            pos: position of the electron

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

        Returns: values of V * psi
        '''
        raise NotImplementedError()


    def jacobian_opt(self,param,pos):

        '''Gradient of the wf wrt the variational parameters 
        at current positions. '''
        return np.array([egrad(self.value,0)(param,p)[0].tolist() for p in pos])
        #return egrad(self.value,0)(parameters,pos)

    def kinetic(self,param,pos):
        '''Compute the action of the kinetic operator on the we points.
        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * psi
        '''
        # value_partial = partial(self.value,param)
        # eg = egrad(egrad(value_partial))(pos)

        eg = egrad(egrad(self.value,1),1)(param,pos)

        if self.ndim == 1:
            return eg.reshape(-1,1)
        else :
            return np.sum(eg,1).reshape(-1,1)

    def applyK(self,param,pos):
        '''Comute the result of H * psi

        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * pis
        ''' 
        Kpsi = -0.5*self.kinetic(param,pos) 
        return Kpsi

    def local_energy(self,param, pos):
        ''' local energy of the sampling points.'''
        return self.applyK(param,pos)/self.value(param, pos) \
               + self.nuclear_potential(pos)  \
               + self.electronic_potential(pos)

    def energy(self,param,pos):
        '''Total energy for the sampling points.'''
        return np.mean(self.local_energy(param,pos))

    def variance(self,param, pos):
        '''Variance of the energy at the sampling points.'''
        return np.var(self.local_energy(param,pos))

    def pdf(self,param,pos):
        '''density of the wave function.'''
        return self.value(param,pos)**2

    def energy_gradient(self,param,pos):
        '''Gradient of the total energy wrt the variational parameters 
        at the current sampling points.'''

        grad_psi = self.jacobian_opt(param,pos)
        psi0 = self.value(param,pos)
        eL = self.local_energy(param,pos)

        if len(param==1):

            grad_psi = grad_psi.reshape(-1,1)
            psi0 = psi0.reshape(-1,1)
            eL = eL.reshape(-1,1)

        G = []
        for i in range(len(param)):
            gi = grad_psi[:,i].reshape(-1,1)
            ratio = gi / psi0
            gg = 2 * ( np.mean(ratio*eL) - np.mean(ratio)*np.mean(eL) )
            G.append( gg )


        return np.array(G)




