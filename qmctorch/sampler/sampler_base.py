import torch
from typing import Dict, Callable
from .. import log
from .walkers import Walkers


class SamplerBase:
    def __init__(
        self,
        nwalkers: int,
        nstep: int,
        step_size: float,
        ntherm: int,
        ndecor: int,
        nelec: int,
        ndim: int,
        init: Dict,
        cuda: bool,
    ) -> None:
        """Base class for the sampler

        Args:
            nwalkers (int): number of walkers
            nstep (int): number of MC steps
            step_size (float): size of the steps in bohr
            ntherm (int): number of MC steps to thermalize
            ndecor (int): number of MC steps to decorellate
            nelec (int): number of electrons in the system
            ndim (int): number of cartesian dimension
            init (dict): method to initialize the walkers
            cuda (bool): turn CUDA ON/OFF
        """

        # self.nwalkers = nwalkers
        self.nelec = nelec
        self.ndim = ndim
        self.nstep = nstep
        self.step_size = step_size
        self.ntherm = ntherm
        self.ndecor = ndecor
        self.cuda = cuda
        if cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.walkers = Walkers(
            nwalkers=nwalkers,
            nelec=nelec,
            ndim=ndim,
            init=init,
            cuda=cuda,
        )

        log.info("")
        log.info(" Monte-Carlo Sampler")
        log.info("  Number of walkers   : {0}", self.walkers.nwalkers)
        log.info("  Number of steps     : {0}", self.nstep)
        log.info("  Step size           : {0}", self.step_size)
        log.info("  Thermalization steps: {0}", self.ntherm)
        log.info("  Decorelation steps  : {0}", self.ndecor)
        log.info("  Walkers init pos    : {0}", init["method"])

    def __call__(
        self, pdf: Callable[[torch.Tensor], torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        """
        Evaluate the sampling algorithm.

        Args:
            pdf (Callable[[torch.Tensor], torch.Tensor]): the function to sample
            *args: additional positional arguments
            **kwargs: additional keyword arguments

        Returns:
            torch.Tensor: the samples
        """
        raise NotImplementedError("Sampler must have a __call__ method")

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + " sampler with  %d walkers" % self.walkers.nwalkers
        )

    def get_sampling_size(self) -> int:
        """evaluate the number of sampling point we'll have."""
        if self.ntherm == -1:
            return self.walkers.nwalkers
        else:
            return self.walkers.nwalkers * int((self.nstep - self.ntherm) / self.ndecor)
