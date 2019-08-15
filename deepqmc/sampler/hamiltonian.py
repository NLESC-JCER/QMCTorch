import torch
from tqdm import tqdm
import time
from torch.autograd import grad, Variable
from deepqmc.sampler.sampler_base import SamplerBase

class Hamiltonian(SamplerBase):

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

        SamplerBase.__init__(self,nwalkers,nstep,nelec,ndim,step_size,domain, move)
        self.traj_length = L

       
    @staticmethod
    def get_grad(func,inp):
        '''Compute the gradient of the of a function
        wrt the value of its input

        Args:   
            func : function to get the gradient of
            inp : input
        Returns:
            grad : gradient of the func wrt the input
        '''

        inp.requires_grad = True
        val = func(inp)
        z = Variable(torch.ones(val.shape))
        val.backward(z)
        grad = inp.grad.data
        inp.grad.data.zero_() 
        inp.requires_grad = False
        return grad

    @staticmethod
    def log_func(func):
        return lambda x: -torch.log(func(x))

    def generate(self,pdf,ntherm=10,with_tqdm=True,pos=None,init='uniform'):

        '''perform a HMC sampling of the pdf
        Returns:
            X (list) : positions of the walkers
        '''

        if ntherm < 0:
            ntherm = self.nstep+ntherm

        self.walkers.initialize(method=init,pos=pos)
        self.walkers.pos = torch.tensor(self.walkers.pos).float()

        # get the logpdf function
        logpdf = self.log_func(pdf)

        POS = []
        rate = 0

        if with_tqdm:
            rng = tqdm(range(self.nstep))
        else:
            rng = range(self.nstep)

        for istep in rng:

            # move the walkers
            self.walkers.pos, _r = self._step(logpdf, self.get_grad, self.step_size, self.traj_length, self.walkers.pos)
            rate += _r

            # store
            if istep>=ntherm:
                POS.append(self.walkers.pos.detach().numpy())

        #print stats
        print("Acceptance rate %1.3f %%" % (rate/self.nstep*100) )
        return POS

    @staticmethod
    def _step(U,get_grad,epsilon,L,qinit):

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
        Einit = U(q) + 0.5 * (p**2).sum(1)

        # half step in momentum space
        p -= 0.5 * epsilon * get_grad(U,q)

        # full steps in q and p space
        for iL in range(L-1):
            q = q + epsilon * p
            p -= 0.5 * epsilon * get_grad(U,q)

        # last full step in pos space
        q = q + epsilon * p

        # half step in momentum space
        p -= 0.5 * epsilon * get_grad(U,q)

        # negate momentum
        p = -p

        # current energy term
        Enew = U(q) + 0.5*(p**2).sum(1)

        # metropolix accept/reject
        eps = torch.rand(Enew.shape)
        cond = (torch.exp(Einit-Enew) < eps).view(-1)
        q[cond] = qinit[cond]

        # comute the accept rate
        rate = cond.sum().float()/cond.shape[0]

        return q, rate
        