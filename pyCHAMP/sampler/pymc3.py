import numpy as np 
import pymc3 as pm
from pyCHAMP.sampler.sampler_base import SAMPLER_BASE

class PYMC3(SAMPLER_BASE):

    def __init__(self, nwalkers=1000, ndim=3):

        ''' Wrapper around the pymc3 samplers 
        Args:
            f (func) : function to sample
            nstep (int) : number of mc step
            nwalkers (int) : number of walkers
            eps (float) : size of the mc step
            boudnary (float) : boudnary of the space
        '''

        SAMPLER_BASE.__init__(self,nwalkers,None,None,None,None,None,None)
        self.ndim = ndim

    def generate(self,pdf):

        ''' perform a MC sampling of the function f
        Returns:
            X (list) : position of the walkers
        '''
        def logp(pos):
            return np.log(pdf(pos))

        with pm.Model() as pot:
            pm.DensityDist('pot', logp=logp, shape=(self.ndim,))

        with pot:
            trace = pm.sample(self.nwalkers, 
                      tune=1000, 
                      target_accept=0.9, 
                      cores=2                      
                      )

        return trace['pot']