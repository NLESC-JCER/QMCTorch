import autograd.numpy as np 
from autograd import elementwise_grad as egrad

class WF(object):

    def __init__(self,nelec, ncart, parameters,derivative='autograd'):

        self.ncart = ncart
        self.nelec = nelec
        self.derivative = derivative
        self.parameters = parameters
        self.ndim = self.nelec*self.ncart
        self.pos = None

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


    def get(self,pos):
        '''Partial to get the value of the wf with current param for given pos.'''
        return partial(self.value,self.parameters)(pos)

    def fp(self,parameters):
        '''Partial to get the value of the wf with current pos for given param.'''
        return partial(self.value,pos=self.pos)(parameters)

    def jacobian_opt(self,parameters):
        '''Gradient of the wf wrt the variational parameters 
        at current positions. '''
        return egrad(self.fp)(parameters)

    def kinetic(self,pos):
        '''Compute the action of the kinetic operator on the we points.
        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * psi
        ''' 
        if self.derivative == 'fd':
            return self._kinetic_fd(pos)
        elif self.derivative == 'autograd':
            return self._kinetic_autograd(pos)
        else:
            raise ValueError('%s is not recognized' %self.derivative)


    def _kinetic_autograd(self,pos):
        eg = egrad(egrad(self.get))(pos)
        if self.ndim == 1:
            return eg.reshape(-1,1)
        else :
            return np.sum(eg,1).reshape(-1,1)

    def _kinetic_fd(self,pos):

        kpsi = np.zeros((pos.shape[0],1))
        for ielec in range(self.nelec):
            for icart in range(self.ncart):
                kpsi += self.d2f_dx2(pos,ielec,icart)

        return - 0.5 * kpsi


    def d2f_dx2(self,pos,ielec,icart):

        eps = 1E-6
        epsm2 = 1./(eps**2)
        index = ielec*self.ncart + icart

        out = -2*self.get(pos)

        pos_tmp = np.copy(pos)
        pos_tmp[:,index] += eps
        out += self.get(pos_tmp)

        pos_tmp = np.copy(pos)
        pos_tmp[:,index] -= eps
        out += self.get(pos_tmp)

        return epsm2 * out

    def applyK(self,pos):
        '''Comute the result of H * psi

        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * pis
        ''' 
        Kpsi = -0.5*self.kinetic(pos) 
        return Kpsi

    def local_energy(self,pos):
        ''' local energy of the sampling points.'''
        return self.applyK(pos)/self.get(pos) \
               + self.nuclear_potential(pos)  \
               + self.electronic_potential(pos)

    def energy(self,pos):
        '''Total energy for the sampling points.'''
        return np.mean(self.local_energy(pos))

    def variance(self,pos):
        '''Variance of the energy at the sampling points.'''
        return np.var(self.local_energy(pos))

    def pdf(self,pos):
        '''density of the wave function.'''
        return self.get(pos)**2

    def energy_gradient(self,parameters,pos):
        '''Gradient of the total energy wrt the variational parameters 
        at the current sampling points.'''

        self.pos = pos
        self.parameters = parameters

        grad_psi = self.jacobian_opt(parameters)
        psi0 = self.get(pos)
        eL = self.local_energy(pos)

        G = []
        for gi in grad_psi:
            ratio = gi / psi0
            G.append(  np.mean(ratio*eL) - np.mean(ratio)*np.mean(eL) )

        return np.array(G)




