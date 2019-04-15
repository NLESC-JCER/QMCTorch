

class WF(object):

    def __init__(self,parameters, nelec, derivative='fd'):

        self.parameters = parameters
        self.nelec = nelec
        self.derivative = derivative

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
        else:
            raise ValueError('%s is not recognized' %self.derivative)


    def _kinetic_fd(self,pos):
        eps = 1E-6
        epsm2 = 1./eps**2
        return epsm2 * ( self.get(pos+eps) - 2*self.get(pos) + self.get(pos-eps) )


    def applyH(self,pos):
        '''Comute the result of H * psi

        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * pis
        ''' 
        return -0.5*self.kinetic(pos) + self.electronic_potential(pos) + self.nuclear_potential(pos) 

    def _density(self,pos):
        return self.get(pos)**2



