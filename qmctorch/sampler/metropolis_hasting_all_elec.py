from tqdm import tqdm
import torch
from time import time
from typing import Callable, Union, Dict
from .sampler_base import SamplerBase
from .. import log

from .proposal_kernels import ConstantVarianceKernel, BaseProposalKernel
from .state_dependent_normal_proposal import StateDependentNormalProposal


class MetropolisHasting(SamplerBase):
    def __init__(
        self,
        kernel: BaseProposalKernel = ConstantVarianceKernel(0.2),
        nwalkers: int = 100,
        nstep: int = 1000,
        ntherm: int = -1,
        ndecor: int = 1,
        nelec: int = 1,
        ndim: int = 3,
        init: Dict = {"min": -5, "max": 5},
        logspace: bool = False,
        cuda: bool = False,
    ) -> None:
        """Metropolis Hasting generator

        Args:
            kernel (BaseProposalKernel, optional): proposal kernel. Defaults to ConstantVarianceKernel(0.2).
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            nstep (int, optional): Number of steps. Defaults to 1000.
            ntherm (int, optional): number of mc step to thermalize. Defaults to -1, i.e. keep ponly last position
            ndecor (int, optional): number of mc step for decorelation. Defauts to 1.
            nelec (int, optional): total number of electrons. Defaults to 1.
            ndim (int, optional): total number of dimension. Defaults to 3.
            init (dict, optional): method to init the positions of the walkers. See Molecule.domain()
            logspace (bool, optional):  Defaults to False.
            cuda (bool, optional): turn CUDA ON/OFF. Defaults to False.

        Returns:
            None

        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> wf = SlaterJastrow(mol)
            >>> sampler = Metropolis(nwalkers=100, nelec=wf.nelec)
            >>> pos = sampler(wf.pdf)
        """

        SamplerBase.__init__(
            self, nwalkers, nstep, 0.0, ntherm, ndecor, nelec, ndim, init, cuda
        )

        self.proposal = StateDependentNormalProposal(kernel, nelec, ndim, self.device)

        self.proposal.kernel.nelec = nelec
        self.proposal.kernel.ndim = ndim

        self.logspace = logspace

        self.log_data()

    def log_data(self) -> None:
        """log data about the sampler."""
        # log.info('  Move type           : {0}', 'all-elec')

    @staticmethod
    def log_func(
        func: Callable[[torch.Tensor], torch.Tensor],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Compute the negative log of  a function

        Args:
            func: input function

        Returns:
            callable: negative log of the function
        """
        return lambda x: torch.log(func(x))

    def __call__(
        self,
        pdf: Callable[[torch.Tensor], torch.Tensor],
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

        with torch.no_grad():
            if self.ntherm < 0:
                self.ntherm = self.nstep + self.ntherm

            self.walkers.initialize(pos=pos)
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
                Xn = self.walkers.pos + self.proposal(self.walkers.pos)

                # new function
                fxn = pdf(Xn)

                # proba ratio
                prob_ratio = fxn / fx

                # get transition ratio
                trans_ratio = self.proposal.get_transition_ratio(self.walkers.pos, Xn)

                # get the proba
                df = prob_ratio * trans_ratio

                # accept the moves
                index = self.accept_reject(df)

                # acceptance rate
                rate += index.byte().sum().float().to("cpu") / (self.walkers.nwalkers)

                # update position/function value
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

    def accept_reject(self, proba: torch.Tensor) -> torch.Tensor:
        """accept the move or not

        Args:
            proba (torch.tensor): probability of each move

        Returns:
            torch.tensor: the indx of the accepted moves
        """

        proba[proba > 1] = 1.0
        tau = torch.rand_like(proba)
        index = (proba - tau >= 0).reshape(-1)
        return index.type(torch.bool)
