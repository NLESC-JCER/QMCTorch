import autograd.numpy as np 
from autograd import elementwise_grad as egrad

class WF(object):

    def __init__(self,nelec, ncart, parameters, derivative='autograd'):

        self.ncart = ncart
        self.parameters = parameters
        self.nelec = nelec
        self.derivative = derivative

        self.ndim = self.nelec*self.ncart

    def get(self,pos):
        ''' Compute the value of the wave function.

        Args:
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
        #Hpsi += self.electronic_potential(pos) 
        #Hpsi += self.nuclear_potential(pos) 
        return Kpsi

    def pdf(self,pos):
        return self.get(pos)**2



