import autograd.numpy as np 
from pyCHAMP.sampler.sampler_base import SAMPLER_BASE
from autograd import elementwise_grad as egrad
from pyhmc import hmc

class HAMILTONIAN(SAMPLER_BASE):

    def __init__(self, nwalkers=1000, nstep=None, nelec=1, ndim=3,
             step_size = None, domain = None,
             move='all'):

        ''' HMC SAMPLER
        Args:
            f (func) : function to sample
            nstep (int) : number of mc step
            nwalkers (int) : number of walkers
            eps (float) : size of the mc step
            boudnary (float) : boudnary of the space
        '''

        SAMPLER_BASE.__init__(self,nwalkers,nstep,nelec,ndim,step_size,domain,move)
        self.nwalkers = nwalkers

    def generate(self,func):

        def logprob(pos,func):
            f_logp = lambda x: np.log(func(x))
            logp = f_logp(pos)
            grad = egrad(f_logp)(pos)

            return logp, grad

        return hmc(logprob, x0=np.random.randn(self.nelec*self.ndim),
                   args=(func,),
                   n_samples=self.nwalkers,
                   epsilon=1,
                   n_burn=int(self.nstep/10))

    # def generate(self,pdf):

    #     niter = 100
    #     h = 0.01
    #     N = 100

    #     tau = la.inv(sigma)

    #     orbit = np.zeros((niter+1, 2))
    #     u = np.array([-3,3])
    #     orbit[0] = u

    #     for k in range(niter):
    #         v0 = np.random.normal(0,1,2)
    #         u, v = leapfrog(tau, u, v0, h, N)

    #         # accept-reject
    #         u0 = orbit[k]
    #         a = np.exp(E(A, u0, v0, u, v))
    #         r = np.random.rand()

    #         if r < a:
    #             orbit[k+1] = u
    #         else:
    #             orbit[k+1] = u0



    # def E(A, u0, v0, u, v):
    #     """Total energy."""
    #     return (u0 @ tau @ u0 + v0 @ v0) - (u @ tau @ u + v @ v)

    # def leapfrog(A, u, v, h, N):

    #     """Leapfrog finite difference scheme."""
    #     v = v - h/2 * A @ u
    #     for i in range(N-1):
    #         u = u + h * v
    #         v = v - h * A @ u

    #     u = u + h * v
    #     v = v - h/2 * A @ u

    #     return u, v