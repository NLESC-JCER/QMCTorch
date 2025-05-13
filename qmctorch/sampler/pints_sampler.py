import torch
import pints
import numpy
from typing import Callable, Union, Dict, Tuple
from .sampler_base import SamplerBase


class torch_model(pints.LogPDF):
    def __init__(self, pdf: Callable[[torch.Tensor], torch.Tensor], ndim: int) -> None:
        """Ancillary class that wraps the wave function in a PINTS class

        Args:
            pdf: wf.pdf function
            ndim: number of dimensions
        """
        self.pdf = pdf
        self.ndim = ndim

    def __call__(self, x: numpy.ndarray) -> numpy.ndarray:
        """Evaluate the log pdf of the wave function at points x

        Args:
            x: positions of the walkers (numpy array)

        Returns:
            values of the log pdf at those points (numpy array)
        """
        x = torch.as_tensor(x).view(1, -1)
        return torch.log(self.pdf(x)).cpu().detach().numpy()

    def evaluateS1(self, x: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Evaluate the log pdf and the gradients of the log pdf at points x

        Args:
            x (numpy.ndarray): positions of the walkers

        Returns:
            tuple:
                log_pdf (numpy.ndarray): values of the log pdf
                grad_log_pdf (numpy.ndarray): gradients of the log pdf
        """

        x = torch.as_tensor(x).view(1, -1)
        pdf = self.pdf(x)
        log_pdf = torch.log(pdf)
        x.requires_grad = True
        grad_log_pdf = 1.0 / pdf * self.pdf(x, return_grad=True)
        return (log_pdf.cpu().detach().numpy(), grad_log_pdf.cpu().detach().numpy())

    def n_parameters(self) -> int:
        """Returns the number of dimensions."""
        return self.ndim


class PintsSampler(SamplerBase):
    def __init__(
        self,
        nwalkers: int = 100,
        method=pints.MetropolisRandomWalkMCMC,
        method_requires_grad=False,
        nstep: int = 1000,
        ntherm: int = -1,
        ndecor: int = 1,
        nelec: int = 1,
        ndim: int = 3,
        init: Dict = {"min": -5, "max": 5},
        cuda: bool = False,
        log_to_screen=False,
        message_interval=20,
    ):
        """Interface to the PINTS Sampler generator

        Args:
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            nstep (int, optional): Number of steps. Defaults to 1000.
            step_size (int, optional): length of the step. Defaults to 0.2.
            nelec (int, optional): total number of electrons. Defaults to 1.
            ntherm (int, optional): number of mc step to thermalize. Defaults to -1, i.e. keep ponly last position
            ndecor (int, optional): number of mc step for decorelation. Defauts to 1.
            ndim (int, optional): total number of dimension. Defaults to 3.
            init (dict, optional): method to init the positions of the walkers. See Molecule.domain()

            move (dict, optional): method to move the electrons. default('all-elec','normal') \n
                                   'type':
                                        'one-elec': move a single electron per iteration \n
                                        'all-elec': move all electrons at the same time \n
                                        'all-elec-iter': move all electrons by iterating through single elec moves \n
                                    'proba' :
                                        'uniform': uniform ina cube \n
                                        'normal': gussian in a sphere \n
            cuda (bool, optional): turn CUDA ON/OFF. Defaults to False.


        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> wf = SlaterJastrow(mol)
            >>> sampler = Metropolis(nwalkers=100, nelec=wf.nelec)
            >>> pos = sampler(wf.pdf)
        """

        SamplerBase.__init__(
            self, nwalkers, nstep, None, ntherm, ndecor, nelec, ndim, init, cuda
        )

        self.method = method
        self.method_requires_grad = method_requires_grad
        self.log_to_screen = log_to_screen
        self.message_interval = message_interval
        self.log_data()

    def log_data(self):
        """log data about the sampler."""
        # log.info(
        #     '  Sampler             : {0}', self.method.name(None))

    @staticmethod
    def log_func(
        func: Callable[[torch.Tensor], torch.Tensor]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Compute the negative log of a function

        Args:
            func (Callable[[torch.Tensor], torch.Tensor]): input function

        Returns:
            Callable[[torch.Tensor], torch.Tensor]: negative log of the function
        """
        return lambda x: torch.log(func(torch.as_tensor(x)))

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

        grad_method = torch.no_grad()
        if self.method_requires_grad:
            grad_method = torch.enable_grad()

        with grad_method:
            if self.ntherm < 0:
                self.ntherm = self.nstep + self.ntherm

            self.walkers.initialize(pos=pos)
            log_pdf = torch_model(pdf, self.walkers.pos.shape[1])

            mcmc = pints.MCMCController(
                log_pdf,
                self.walkers.nwalkers,
                self.walkers.pos.cpu(),
                method=self.method,
            )
            mcmc.set_max_iterations(self.nstep)
            mcmc._log_to_screen = self.log_to_screen
            mcmc._message_interval = self.message_interval
            chains = mcmc.run()

        chains = chains[:, self.ntherm :: self.ndecor, :]
        chains = chains.reshape(-1, self.nelec * self.ndim)
        return torch.as_tensor(chains).requires_grad_()
