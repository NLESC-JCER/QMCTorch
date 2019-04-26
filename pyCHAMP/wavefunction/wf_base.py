import autograd.numpy as np 
from autograd import elementwise_grad as egrad
from autograd import hessian, jacobian
from functools import partial

class WF(object):

    def __init__(self,nelec, ndim):

        self.ndim = ndim
        self.nelec = nelec
        self.ndim_tot = self.nelec*self.ndim
        self.eps = 1E-6

    def values(self,parameters,pos):
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

        Returns: values of V * psi
        '''
        raise NotImplementedError()


    def jacobian_opt(self,param,pos,normalized=False):

        '''Gradient of the wf wrt the variational parameters 
        at current positions. '''
        jac = np.array([egrad(self.values,0)(param,p.reshape(1,-1))[0].tolist() for p in pos])

        if jac.ndim == 1:
            jac = jac.reshape(-1,1)
        return jac
        

    def kinetic_egrad(self,param,pos):
        
        '''Compute the action of the kinetic operator on the we points.
        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * psi
        '''
        eg = egrad(egrad(self.values,1),1)(param,pos)

        if self.ndim_tot == 1:
            return eg.reshape(-1,1)
        else :
            return np.sum(eg,1).reshape(-1,1)

    def kinetic_hess(self,param,pos):
        '''Compute the action of the kinetic operator on the we points.
        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * psi
        '''
        h = hessian(self._one_value,1)
        eg = np.array([np.diag(h(param,p)) for p in pos])

        if self.ndim_tot == 1:
            return eg.reshape(-1,1)
        else :
            return np.sum(eg,1).reshape(-1,1)

    def kinetic_fd(self,param,pos,eps=1E-6):

        '''Compute the action of the kinetic operator on the we points.
        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * psi
        '''

        ndim = pos.shape[1]
        out = np.zeros_like(pos)

        for icol in range(ndim):

            pos_tmp = np.copy(pos)
            feps = -2*self.values(param,pos_tmp)

            pos_tmp = np.copy(pos)
            pos_tmp[:,icol] += eps
            feps += self.values(param,pos_tmp)

            pos_tmp = np.copy(pos)
            pos_tmp[:,icol] -= eps
            feps += self.values(param,pos_tmp)

            out[:,icol] = feps.reshape(-1)/(eps**2)

        if self.ndim_tot == 1:
            return out.reshape(-1,1)
        else :
            return np.sum(out,1).reshape(-1,1)

    def drift_fd(self,param,pos,eps=1E-6):

        '''Compute the drift force on the points.
        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of Grad Psi
        '''

        ndim = pos.shape[1]
        out = np.zeros_like(pos)

        for icol in range(ndim):

            pos_tmp = np.copy(pos)
            pos_tmp[:,icol] += eps
            feps = self.values(param,pos_tmp)

            pos_tmp = np.copy(pos)
            pos_tmp[:,icol] -= eps
            feps -= self.values(param,pos_tmp)

            out[:,icol] = feps.reshape(-1)/(2*eps)

        if self.ndim_tot == 1:
            return 2*out.reshape(-1,1)/self.values(param,pos)
        else :
            return 2*out/self.values(param,pos)


    def applyK(self,param,pos):
        '''Comute the result of H * psi

        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * pis
        ''' 
        Kpsi = -0.5*self.kinetic_fd(param,pos) 
        return Kpsi

    
    def local_energy(self,param, pos):
        ''' local energy of the sampling points.'''
        return self.applyK(param,pos)/self.values(param, pos) \
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
        vals = self.values(param,pos)**2
        #vals[vals==0] += self.eps
        return vals

    def auto_energy_gradient(self,param,pos):
        return egrad(self.energy,0)(param,pos)

    def energy_gradient(self,param,pos):
        '''Gradient of the total energy wrt the variational parameters 
        at the current sampling points.'''

        grad_psi = self.jacobian_opt(param,pos)
        psi0 = self.values(param,pos)
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




