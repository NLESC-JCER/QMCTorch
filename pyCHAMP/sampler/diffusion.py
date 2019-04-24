import numpy as np
from pyCHAMP.sampler.sampler_base import SAMPLER_BASE

class DIFFUSION(SAMPLER_BASE):

    def __init__(self, nwalkers=1000, nstep=1000, nelec=1, ndim=3, 
                 step_size = 0.1, domain = {'min':-2,'max':2},
                 move='one'):

        ''' Diffusion  SAMPLER
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

    def generate(self,func_el):

        self.walkers.initialize(method='center')
        Ex = func_el(self.walkers.pos)
        E = np.mean(fx)

        for istep in range(nstep):

            # new positions
            Xn = self.walkers.move(self.step_size,method=self.move)

            # new local energies
            Exn = func_el(Xn)

            # update the walkers
            self._update_walkers(Exn,Ex,E)

            # update total energy
            Ex = func_el(self.walkers.pos)
            E = np.mean(Ex)

        return self.walkers.pos

    def _update_walkers(self,Exn,Ex,E):

        # number of each walker
        N = np.exp(0.5*step_size*(Ex+Exn-2*E)).astype('int')

        # kill the walkers
        self.walkers.status[N==0] = 0

        # multply walkers
        index = N>1
        num_new = np.sum(index)
        self.walkers.nwalkers += num_new
        self.walkers.status = np.stack(self.walkers.status,np.ones(num_new))
        self.walkers.pos = np.vstack(self.walkers.pos,self.walker.pos[index,:])
        













