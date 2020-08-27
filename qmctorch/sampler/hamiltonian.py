import torch
from tqdm import tqdm
from torch.autograd import Variable
from .sampler_base import SamplerBase
from .. import log


class Hamiltonian(SamplerBase):

    def __init__(self, mol, nwalkers, ntherm,
                 nstep=None,
                 nsample=None, ndecor=None,
                 step_size=0.1,
                 init='atomic', L=10,
                 cuda=False):
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

        SamplerBase.__init__(self, mol, nwalkers, nsample, nstep,
                             step_size, ntherm, ndecor, init, cuda)
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
    def log_func(pdf):
        return lambda x: -torch.log(pdf(x))

    def grad_log_func(self, logpdf):
        return lambda x: self.get_grad(logpdf, x)
        # return lambda x: pdf(x, return_grad=True)/pdf(x).unsqueeze(-1)

    def __call__(self, pdf, pos=None, with_tqdm=True):
        """Generate walkers following HMC

        Arguments:
            pdf {callable} -- density to sample
            pos (torch.tensor): precalculated position to start with
            with_tqdm (bool, optional): use tqdm progress bar. Defaults to True.

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

        # get the gradients of the log pdf
        grad_logpdf = self.grad_log_func(logpdf)

        pos = []
        rate = 0
        idecor = 0

        rng = tqdm(range(self.nstep),
                   desc='INFO:QMCTorch|  HMC Sampling',
                   disable=not with_tqdm)

        for istep in rng:

            # move the walkers
            self.walkers.pos, _r = self._step(
                logpdf, grad_logpdf, self.step_size, self.traj_length,
                self.walkers.pos)
            rate += _r

            # store
            if ((istep+1) >= self.ntherm):
                if (idecor % self.ndecor == 0):
                    pos.append(self.walkers.pos.detach())
                idecor += 1

        # print stats
        log.options(style='percent').debug(
            "  Acceptance rate %1.3f %%" % (rate / self.nstep * 100))
        return torch.cat(pos).requires_grad_()

    @staticmethod
    def _step(U, gradU, epsilon, L, qinit):
        '''Propagates all the walkers over on traj
        Args:
            U (callable): the target pdf
            get_grad (callable) : get the value of the target dist gradient
            epsilon (float) : step size
            L (int) : number of steps in the traj
            qinit (array) : initial positon of the walkers
        Returns:
            q : new positions of the walkers
            rate : accept rate
        '''

        # init the momentum
        q = qinit.clone().requires_grad_()
        p = torch.randn(q.shape)

        # initial energy terms
        Einit = U(q) + 0.5 * (p*p).sum(1)

        # half step in momentum space
        p -= 0.5 * epsilon * gradU(q)

        # full steps in q and p space
        for iL in range(L - 1):
            q = q + epsilon * p
            p -= 0.5 * epsilon * gradU(q)

        # last full step in pos space
        q = q + epsilon * p

        # half step in momentum space
        p -= 0.5 * epsilon * gradU(q)

        # negate momentum
        p = -p

        # current energy term
        Enew = U(q) + 0.5 * (p*p).sum(1)

        # metropolix accept/reject
        eps = torch.rand(Enew.shape)
        cond = (torch.exp(Einit - Enew) < eps).view(-1)
        q[cond] = qinit[cond]

        # comute the accept rate
        rate = cond.sum().float() / cond.shape[0]

        return q, rate
