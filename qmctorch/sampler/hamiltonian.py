from typing import Dict, Callable, Optional, Tuple

import torch
from tqdm import tqdm

from .sampler_base import SamplerBase
from .. import log


class Hamiltonian(SamplerBase):
    def __init__(
        self,
        nwalkers: int = 100,
        nstep: int = 100,
        step_size: float = 0.2,
        L: int = 10,
        ntherm: int = -1,
        ndecor: int = 1,
        nelec: int = 1,
        ndim: int = 3,
        init: Dict = {"min": -5, "max": 5},
        cuda: bool = False,
    ):
        """Hamiltonian Monte Carlo Sampler.

        Args:
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            nstep (int, optional): Number of steps. Defaults to 100.
            step_size (int, optional): length of the step. Defaults to 0.2.
            L (int, optional): length of the trajectory . Defaults to 10.
            nelec (int, optional): total number of electrons. Defaults to 1.
            ntherm (int, optional): number of mc step to thermalize. Defaults to -1, i.e. keep only last position
            ndecor (int, optional): number of mc step for decorrelation. Defaults to 1.
            ndim (int, optional): total number of dimension. Defaults to 3.
            init (dict, optional): method to init the positions of the walkers. See Molecule.domain()
            cuda (bool, optional): turn CUDA ON/OFF. Defaults to False.
        """

        SamplerBase.__init__(
            self, nwalkers, nstep, step_size, ntherm, ndecor, nelec, ndim, init, cuda
        )
        self.traj_length = L

    @staticmethod
    def get_grad(
        func: Callable[[torch.Tensor], torch.Tensor], inp: torch.Tensor
    ) -> torch.Tensor:
        """get the gradient of the pdf using autograd

        Args:
            func: function to compute the pdf
            inp: input of the function

        Returns:
            gradient of the wavefunction
        """
        with torch.enable_grad():
            if inp.grad is not None:
                inp.grad.zero_()
            inp.requires_grad = True

            val = func(inp)
            val.backward(torch.ones(val.shape))

            inp.requires_grad = False
        return inp.grad

    @staticmethod
    def log_func(func: Callable[[torch.Tensor], torch.Tensor]):
        """Compute the negative log of  a function

        Args:
            func (callable): input function

        Returns:
            callable: negative log of the function
        """
        return lambda x: -torch.log(func(x))

    def __call__(
        self,
        pdf: Callable[[torch.Tensor], torch.Tensor],
        pos: Optional[torch.Tensor] = None,
        with_tqdm: bool = True,
    ) -> torch.Tensor:
        """Generate walkers following HMC

        Generates a series of walkers following the HMC algorithm

        Args:
            pdf (callable): density to sample
            pos (torch.tensor, optional): precalculated position to start with.
                                          Defaults to None.
            with_tqdm (bool, optional): use tqdm progress bar. Defaults to True.

        Returns:
            torch.tensor: sampling points
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

        rng = tqdm(
            range(self.nstep),
            desc="INFO:QMCTorch|  Sampling",
            disable=not with_tqdm,
        )

        for istep in rng:
            # move the walkers
            self.walkers.pos, _r = self._step(
                logpdf,
                self.get_grad,
                self.step_size,
                self.traj_length,
                self.walkers.pos,
            )
            rate += _r

            # store
            if istep >= self.ntherm:
                if idecor % self.ndecor == 0:
                    pos.append(self.walkers.pos)
                idecor += 1

        # print stats
        log.options(style="percent").debug(
            "  Acceptance rate %1.3f %%" % (rate / self.nstep * 100)
        )
        return torch.cat(pos).requires_grad_()

    @staticmethod
    def _step(
        U: Callable[[torch.Tensor], torch.Tensor],
        get_grad: Callable[
            [Callable[[torch.Tensor], torch.Tensor], torch.Tensor], torch.Tensor
        ],
        epsilon: float,
        L: int,
        q_init: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Take one step of the sampler

        Args:
            U (Callable[[torch.Tensor], torch.Tensor]): The target pdf
            get_grad (Callable[[Callable[[torch.Tensor], torch.Tensor], torch.Tensor], torch.Tensor]): Function to get the gradient of the target distribution
            epsilon (float): Step size
            L (int): Number of steps in the trajectory
            q_init (torch.Tensor): Initial position of the walkers

        Returns:
            Tuple[torch.Tensor, float]: Updated positions and acceptance rate
        """

        q = q_init.clone()

        # Initialize the momentum
        p = torch.randn(q.shape)

        # Initial energy terms
        E_init = U(q) + 0.5 * (p * p).sum(1)

        # Half step in momentum space
        p -= 0.5 * epsilon * get_grad(U, q)

        # Full steps in q and p space
        for _ in range(L - 1):
            q += epsilon * p
            p -= epsilon * get_grad(U, q)

        # Last full step in position space
        q += epsilon * p

        # Half step in momentum space
        p -= 0.5 * epsilon * get_grad(U, q)

        # Negate momentum
        p = -p

        # Current energy term
        E_new = U(q) + 0.5 * (p * p).sum(1)

        # Metropolis accept/reject
        eps = torch.rand(E_new.shape)
        rejected = torch.exp(E_init - E_new) < eps
        q[rejected] = q_init[rejected]

        # Compute the acceptance rate
        rate = 1 - rejected.sum().float() / rejected.shape[0]

        return q, rate
