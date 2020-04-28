import torch
from tqdm import tqdm
from torch.autograd import Variable
from .sampler_base import SamplerBase


class Hamiltonian(SamplerBase):

    def __init__(self, nwalkers=100, nstep=100, nelec=1, ndim=3,
                 step_size=0.1, ntherm=-1, ndecor=1, init={'min': -2, 'max': 2}, L=10,
                 with_tqdm=True):
        """Hamiltonian Monte Carlo Sampler.

        Args:
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            nstep (int, optional): Number of steps. Defaults to 1000.
            step_size (int, optional): length of the step. Defaults to 3.
            nelec (int, optional): total number of electrons. Defaults to 1.
            ntherm (int, optional): number of mc step to thermalize. Defaults to -1, i.e. keep ponly last position
            ndecor (int, optional): number of mc step for decorelation. Defauts to 1.
            ndim (int, optional): total number of dimension. Defaults to 1.
            init (dict, optional): method to init the positions of the walkers. See Molecule.domain()
            L (int, optional): length of the trajectory . Defaults to 10.
            with_tqdm (bool, optional): use tqdm progress bar. Defaults to True.
        """

        SamplerBase.__init__(self, nwalkers, nstep,
                             step_size, ntherm, ndecor,
                             nelec, ndim, init, with_tqdm)
        self.traj_length = L

    @staticmethod
    def get_grad(func, inp):
        inp.requires_grad = True
        val = func(inp)
        z = Variable(torch.ones(val.shape))
        val.backward(z)
        fgrad = inp.grad.data
        inp.grad.data.zero_()
        inp.requires_grad = False
        return fgrad

    @staticmethod
    def log_func(func):
        return lambda x: -torch.log(func(x))

    def __call__(self, pdf, pos=None):
        """Generate walkers followinf HMC

        Arguments:
            pdf {callable} -- density to sample

        Keyword Arguments:
            ntherm {int} -- number of iterations needed to thermalize (default: {10})
            ndecor {int} -- number of iterations needed to decorelate (default: {10})
            pos {torch.tensor} -- initial position of the walker (default: {None})

        Returns:
            torch.tensor -- sampling points
        """

        if self.ntherm < 0:
            self.ntherm = self.nstep + self.ntherm

        self.walkers.initialize(pos=pos)
        self.walkers.pos = self.walkers.pos.clone()

        # get the logpdf function
        logpdf = self.log_func(pdf)

        pos = []
        rate = 0
        idecor = 0

        if self.with_tqdm:
            rng = tqdm(range(self.nstep))
        else:
            rng = range(self.nstep)

        for istep in rng:

            # move the walkers
            self.walkers.pos, _r = self._step(
                logpdf, self.get_grad, self.step_size, self.traj_length,
                self.walkers.pos)
            rate += _r

            # store
            if (istep >= self.ntherm):
                if (idecor % self.ndecor == 0):
                    pos.append(self.walkers.pos.detach())
                idecor += 1

        # print stats
        print("Acceptance rate %1.3f %%" % (rate / self.nstep * 100))
        return torch.cat(pos).requires_grad_()

    @staticmethod
    def _step(U, get_grad, epsilon, L, qinit):
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
        p -= 0.5 * epsilon * get_grad(U, q)

        # full steps in q and p space
        for iL in range(L - 1):
            q = q + epsilon * p
            p -= 0.5 * epsilon * get_grad(U, q)

        # last full step in pos space
        q = q + epsilon * p

        # half step in momentum space
        p -= 0.5 * epsilon * get_grad(U, q)

        # negate momentum
        p = -p

        # current energy term
        Enew = U(q) + 0.5 * (p**2).sum(1)

        # metropolix accept/reject
        eps = torch.rand(Enew.shape)
        cond = (torch.exp(Einit - Enew) < eps).view(-1)
        q[cond] = qinit[cond]

        # comute the accept rate
        rate = cond.sum().float() / cond.shape[0]

        return q, rate
