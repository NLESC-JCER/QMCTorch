import torch
from tqdm import tqdm
import time
from torch.autograd import grad, Variable
from pyCHAMP.sampler.sampler_base import SAMPLER_BASE


class HAMILTONIAN_TORCH(SAMPLER_BASE):

    def __init__(self, nwalkers=1000, nstep=None, nelec=1, ndim=3,
             step_size = None, domain = None, L=10,
             move='all'):

        ''' HMC SAMPLER
        Args:
            f (func) : function to sample
            nstep (int) : number of mc step
            nwalkers (int) : number of walkers
            eps (float) : size of the mc step
            boudnary (float) : boudnary of the space
        '''

        SAMPLER_BASE.__init__(self,nwalkers,nstep,nelec,ndim,step_size,domain, move)
        self.traj_length = L

       
    # @staticmethod
    # def get_grad(pdf,pos):
    #     '''Compute the gradient of the density.'''
    #     poscp = pos.clone()
    #     poscp.requires_grad = True
    #     val = pdf(poscp)
    #     z = Variable(torch.ones(val.shape))
    #     return grad(val,poscp,grad_outputs=z,create_graph=False)[0]

    @staticmethod
    def get_grad(pdf,pos):
        '''Compute the gradient of the density.'''

        # poscp = pos.clone()
        # poscp.requires_grad = True
        pos.requires_grad = True
        val = pdf(pos)
        z = Variable(torch.ones(val.shape))
        val.backward(z)
        grad = pos.grad.data
        pos.grad.data.zero_() 
        pos.requires_grad = False
        return grad



    def generate(self,pdf,ntherm=10,with_tqdm=True,pos=None,init='center'):

        '''perform a HMC sampling of the pdf
        Returns:
            X (list) : positions of the walkers
        '''

        if ntherm < 0:
            ntherm = self.nstep+ntherm

        self.walkers.initialize(method=init,pos=pos)
        self.walkers.pos = torch.tensor(self.walkers.pos).float()

        POS = []
        rate = 0

        if with_tqdm:
            rng = tqdm(range(self.nstep))
        else:
            rng = range(self.nstep)

        for istep in rng:

            # move the walkers
            self.walkers.pos, _r = self._step(pdf, self.get_grad, self.step_size, self.traj_length, self.walkers.pos)
            rate += _r

            # store
            if istep>=ntherm:
                POS.append(self.walkers.pos.detach().numpy())

        #print stats
        print("Acceptance rate %1.3f %%" % (rate/self.nstep*100) )
        return POS

    @staticmethod
    def _step(pdf,get_grad,epsilon,L,qinit):

        '''Propagates all the walkers over on traj
        Args:
            pdf (callable): the target dist
            get_grad (callable) : get the value of the target dist gradient
            epsilon (float) : step size
            L (int) : number of steps in the traj
            qinit (array) : initial positon of the walkers
        Returns:
            q : new positions of the walkers
            rate : accept rate  
        '''

        # init the momentum
        q = qinit.clone()
        p = torch.randn(q.shape)

        # initial energy terms
        U = pdf(q)
        K = 0.5 * p.sum(1)

        # half step in momentum space
        p -= 0.5 * epsilon * get_grad(pdf,q)

        # full steps in q and p space
        for iL in range(L-1):
            q = q + epsilon * p
            p -= 0.5 * epsilon * get_grad(pdf,q)

        # last full step in pos space
        q = q + epsilon * p

        # half step in momentum space
        p -= 0.5 * epsilon * get_grad(pdf,q)

        # negate momentum
        p = -p

        # current energy term
        U += pdf(q)
        K += 0.5*p.sum(1)

        # metropolix accept/reject
        cond = (torch.exp(U-K) < 1).view(-1)
        rate = cond.byte().sum().float()/cond.shape[0]
        q[cond] = qinit[cond]

        return q, rate
        



    # def generate(self,func):

    #     def logprob(pos,func):
    #         f_logp = lambda x: np.log(func(x))
    #         logp = f_logp(pos)
    #         grad = egrad(f_logp)(pos)

    #         return logp, grad

    #     return hmc(logprob, x0=np.random.randn(self.nelec*self.ndim),
    #                args=(func,),
    #                n_samples=self.nwalkers,
    #                epsilon=1,
    #                n_burn=int(self.nstep/10))

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