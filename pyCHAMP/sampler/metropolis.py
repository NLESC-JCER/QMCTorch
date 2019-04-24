import numpy as np 
from pyCHAMP.sampler.sampler_base import SAMPLER_BASE
from pyCHAMP.sampler.walkers import WALKERS

class METROPOLIS(SAMPLER_BASE):

    def __init__(self, nwalkers=1000, nstep=1000, nelec=1, ndim=3,
                 step_size = 3, domain = {'min':-2,'max':2},
                 move='all'):

        ''' METROPOLIS HASTING SAMPLER
        Args:
            f (func) : function to sample
            nstep (int) : number of mc step
            nwalkers (int) : number of walkers
            eps (float) : size of the mc step
            boudnary (float) : boudnary of the space
        '''

        self.nwalkers = nwalkers
        self.nstep = nstep
        self.step_size = step_size
        self.domain = domain
        self.move = move

        self.walkers = WALKERS(nwalkers,nelec,ndim,domain)

    def generate(self,pdf):

        ''' perform a MC sampling of the function f
        Returns:
            X (list) : position of the walkers
        '''

        self.walkers.initialize(method='uniform')
        fx = pdf(self.walkers.pos)
        ones = np.ones((self.nwalkers,1))

        for istep in range(self.nstep):

            # new positions
            Xn = self.walkers.move(self.step_size,method=self.move)
            
            # new function
            fxn = pdf(Xn)
            df = fxn/fx

            # accept the moves
            index = self._accept(df)

            # update position/function values
            self.walkers.pos[index,:] = Xn[index,:]
            fx[index] = fxn[index]
        
        return self.walkers.pos

    
    def _accept(self,df):
        ones = np.ones((self.nwalkers,1))
        P = np.minimum(ones,df)
        tau = np.random.rand(self.nwalkers,1)
        return (P-tau>=0).reshape(-1)

