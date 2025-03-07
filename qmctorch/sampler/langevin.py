from tqdm import tqdm
import torch
from torch.distributions import MultivariateNormal
from time import time
from typing import Callable, Union, Dict
from .sampler_base import SamplerBase
from .. import log


class Langevin(SamplerBase):
    def __init__(  # pylint: disable=dangerous-default-value
        self,
        nwalkers: int = 100,
        nstep: int = 1000,
        step_size: float = 0.2,
        ntherm: int = -1,
        ndecor: int = 1,
        nelec: int = 1,
        ndim: int = 3,
        init: Dict = {"min": -5, "max": 5},
        cuda: bool = False,
    ):
        """Metropolis Hasting generator

        Args:
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            nstep (int, optional): Number of steps. Defaults to 1000.
            step_size (int, optional): length of the step. Defaults to 0.2.
            nelec (int, optional): total number of electrons. Defaults to 1.
            ntherm (int, optional): number of mc step to thermalize. Defaults to -1, i.e. keep only the last position
            ndecor (int, optional): number of mc step for decorelation. Defauts to 1.
            ndim (int, optional): total number of dimension. Defaults to 3.
            init (dict, optional): method to init the positions of the walkers. See Molecule.domain()

            move (dict, optional): method to move the electrons. default('all-elec','normal') \n
                                   'type':
                                        'one-elec': move a single electron per iteration \n
                                        'all-elec': move all electrons at the same time \n
                                        'all-elec-iter': move all electrons by iterating through single elec moves \n
                                    'proba' :
                                        'uniform': uniform in a cube \n
                                        'normal': gussian in a sphere \n
            cuda (bool, optional): turn CUDA ON/OFF. Defaults to False.


        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> wf = SlaterJastrow(mol)
            >>> sampler = Metropolis(nwalkers=100, nelec=wf.nelec)
            >>> pos = sampler(wf.pdf)
        """

        SamplerBase.__init__(
            self, nwalkers, nstep, step_size, ntherm, ndecor, nelec, ndim, init, cuda
        )
        self.requires_autograd = True
        self.log_data()

    def log_data(self):
        """log data about the sampler."""
        log.info("  Move type           : {0}", "all-elec")
        log.info("  Move proba          : {0}", "normal")

    @staticmethod
    def log_func(func):
        """Compute the negative log of  a function

        Args:
            func (callable): input function

        Returns:
            callable: negative log of the function
        """
        return lambda x: torch.log(func(x))

    def __call__(
        self,
        pdf: Callable,
        pos: Union[None, torch.Tensor] = None,
        with_tqdm: bool = True,
    ) -> torch.Tensor:
        """Generate a series of point using MC sampling

        Args:
            pdf (callable): probability distribution function to be sampled
            pos (torch.tensor, optional): position to start with.
                                          Defaults to None.
            with_tqdm (bool, optional): use tqdm progress bar. Defaults to True.

        Returns:
            torch.tensor: positions of the walkers
        """

        if self.ntherm >= self.nstep:
            raise ValueError("Thermalisation longer than trajectory")

        if self.ntherm < 0:
            self.ntherm = self.nstep + self.ntherm

        # pdf = self.log_func(pdf)

        self.walkers.initialize(pos=pos)
        self.walkers.pos.requires_grad = True
        fx = pdf(self.walkers.pos)

        pos, rate, idecor = [], 0, 0
        rng = tqdm(
            range(self.nstep),
            desc="INFO:QMCTorch|  Sampling",
            disable=not with_tqdm,
        )
        tstart = time()

        for istep in rng:

            # new positions
            Xn = self.move(pdf)

            # new function
            fxn = pdf(Xn)

            # ratio density
            df = fxn / fx

            # ratio transition 
            q0 = self.transition_probability(self.walkers.pos, Xn, pdf) 
            q1 = self.transition_probability(Xn, self.walkers.pos, pdf)
            dq = q0/q1
            # dq = 1.

            # accept the moves
            index = self.accept(df*dq)

            # acceptance rate
            rate += index.byte().sum().float().to("cpu") / (self.walkers.nwalkers)

            # update position/function value
            with torch.no_grad():
                self.walkers.pos[index, :] = Xn[index, :]
                fx[index] = fxn[index]

            if istep >= self.ntherm:
                if idecor % self.ndecor == 0:
                    pos.append(self.walkers.pos.to("cpu").clone())
                idecor += 1

        if with_tqdm:
            log.info(
                "   Acceptance rate     : {:1.2f} %", (rate / self.nstep * 100)
            )
            log.info(
                "   Timing statistics   : {:1.2f} steps/sec.",
                self.nstep / (time() - tstart),
            )
            log.info("   Total Time          : {:1.2f} sec.", (time() - tstart))

        return torch.cat(pos).requires_grad_()

    def move(self, pdf: Callable) -> torch.Tensor:
        """Move all electron in a vectorized way.

        Args:
            pdf (callable): function to sample
            id_elec (int): index f the electron to move

        Returns:
            torch.tensor: new positions of the walkers
        """ 
        density_values = pdf(self.walkers.pos)
        grads = torch.autograd.grad(density_values, 
                                    self.walkers.pos, 
                                    torch.ones_like(density_values), create_graph=False)[0]
        displacement = torch.randn_like(self.walkers.pos).to(self.device)
        return self.walkers.pos + self.step_size * grads + (2*self.step_size) ** 0.5 * displacement

        
    def transition_probability(self, 
                               Xn: torch.Tensor, 
                               Xo: torch.Tensor, 
                               pdf: Callable) -> torch.Tensor:
        """Transition density between the new and old positions

        Args:
            Xn (torch.tensor): new positions
            Xo (torch.tensor): old positions

        Returns:
            torch.tensor: transition density
        """
        vals = pdf(Xo)
        grad = torch.autograd.grad(vals, Xo, torch.ones_like(vals))[0]
        return torch.exp(-(torch.norm(Xn - Xo - self.step_size*grad, p=2, dim=1)**2) / (4*self.step_size))

    def accept(self, proba: torch.Tensor) -> torch.Tensor:
        """accept the move or not

        Args:
            proba (torch.tensor): probability of each move

        Returns:
            t0rch.tensor: the index of the accepted moves
        """
        proba[proba > 1] = 1.0
        tau = torch.rand_like(proba)
        index = (proba - tau >= 0).reshape(-1)
        return index.type(torch.bool)