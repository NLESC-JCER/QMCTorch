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


    # def generate(self,pdf):

    #     ''' perform a MC sampling of the function f
    #     Returns:
    #         X (list) : position of the walkers
    #     '''
    #     eps = 1E-6

    #     if self.initial_guess is None:
    #         X = self.boundary * (-1 + 2*np.random.rand(self.nwalkers,self.ndim))
    #     else:
    #         X = self.initial_guess

    #     fx = pdf(X)
    #     fx[fx==0] = eps
    #     ones = np.ones((self.nwalkers,1))

    #     for istep in range(self.nstep):

    #         # new position
    #         if self.all_electron_move:
    #             xn =  X + self.mc_step_size * (2*np.random.rand(self.nwalkers,self.ndim) - 1)   
    #         else:
    #             raise ValueError('Single electorn moves not yet implemented')


    #         # new function
    #         fxn = pdf(xn)
    #         df = fxn/(fx)

    #         # probability
    #         P = np.minimum(ones,df)
    #         tau = np.random.rand(self.nwalkers,1)

    #         # update
    #         index = (P-tau>=0).reshape(-1)
    #         X[index,:] = xn[index,:]
    #         fx[index] = fxn[index]
    #         fx[fx==0] = eps
        
    #     return X