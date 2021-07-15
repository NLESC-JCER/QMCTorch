import math
from time import time
from typing import Callable, Union

import torch
from torch.distributions import Normal
from torch.nn import Parameter
from torch.optim import SGD
from tqdm import tqdm

from .sampler_base import SamplerBase
from .. import log
from ..scf import Molecule

sqrt_2pi = math.sqrt(2 * math.pi)


class Rejection(SamplerBase, torch.nn.Module):

    def __init__(self,
                 mol: Molecule,
                 nwalkers: int = 100,
                 nstep: int = 1000,
                 step_size: float = 0.2,
                 ntherm: int = -1,
                 ndecor: int = 1,
                 nelec: int = 1,
                 ndim: int = 3,
                 logspace: bool = False,
                 cuda: bool = False):
        """Metropolis Hasting generator

        Args:
            nstep (int, optional): Number of steps. Defaults to 1000.
            domain (dict, optional): Domain in which to sample, See Molecule.domain()
            nelec (int, optional): total number of electrons. Defaults to 1.
            ndim (int, optional): total number of dimension. Defaults to 3.
            cuda (bool, optional): turn CUDA ON/OFF. Defaults to False.


        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> wf = SlaterJastrow(mol)
            >>> sampler = Rejection(nwalkers=100, nelec=wf.nelec)
            >>> pos = sampler(wf.pdf)
        """

        init = mol.domain('normal')

        torch.nn.Module.__init__(self)
        SamplerBase.__init__(self, nwalkers, nstep,
                             step_size, ntherm, ndecor,
                             nelec, ndim, init, cuda)

        self.mean = Parameter(data=torch.tensor(init['mean'], dtype=torch.float), requires_grad=True)
        self.sigma = Parameter(data=torch.tensor(init['sigma'].diagonal(), dtype=torch.float), requires_grad=True)

        self.logspace = logspace
        self.log_data()

    def log_data(self):
        """log data about the sampler."""
        pass

    @staticmethod
    def norm_pdf(z, m=torch.tensor(0.), s=torch.tensor(1.)):
        return torch.exp(-(z - m) ** 2 / (2 * s ** 2)) / (sqrt_2pi * s)

    def _sample(self):
        return Normal(self.mean, self.sigma).sample([self.nwalkers * self.nelec]).view(self.nwalkers,
                                                                                       self.nelec * self.ndim)

    def __call__(self, pdf: Callable, pos: Union[None, torch.Tensor] = None,
                 with_tqdm: bool = True) -> torch.Tensor:
        """Generate a series of point using MC sampling

        Args:
            pdf (callable): probability distribution function to be sampled
            pos (torch.tensor, optional): position to start with.
                                          Defaults to None.
            with_tqdm (bool, optional): use tqdm progress bar. Defaults to True.

        Returns:
            torch.tensor: positions of the walkers
        """

        _type_ = torch.get_default_dtype()
        if _type_ == torch.float32:
            eps = 1E-7
        elif _type_ == torch.float64:
            eps = 1E-16

        if self.ntherm >= self.nstep:
            raise ValueError('Thermalisation longer than trajectory')
        elif self.ntherm < 0:
            self.ntherm = self.nstep + self.ntherm

        opt = SGD(self.parameters(), lr=self.step_size)

        pos, rate, idecor = [], 0, 0

        self.walkers.pos = self._sample()

        rng = tqdm(range(self.nstep),
                   desc='INFO:QMCTorch|  Sampling',
                   disable=not with_tqdm)
        tstart = time()

        for istep in rng:
            with torch.enable_grad():
                opt.zero_grad()  # zero the gradient buffers

                new_pos = self._sample()

                g = torch.ones(self.nwalkers)
                for i in range(self.ndim):
                    g *= self.norm_pdf(new_pos[:, i], m=self.mean[i], s=self.sigma[i])
                f = pdf(new_pos)

                m = torch.max(f / g)

                u = torch.rand_like(g)
                gum = g * u * m

                # accept the moves
                accepted = gum < f

                self.walkers.pos[accepted] = new_pos[accepted]

                # acceptance rate
                rate += torch.sum(accepted) / self.nwalkers

                loss = torch.mean((g - f) ** 2)
                print("-------------")
                print("mean:", self.mean.data)
                print("sigma:", self.sigma.data)
                print("acc rate:", torch.sum(accepted) / self.nwalkers)
                print("loss:", loss)
                loss.backward()
                opt.step()
                self.sigma.data = torch.clamp(self.sigma, min=eps)

            if (istep >= self.ntherm):
                if (idecor % self.ndecor == 0):
                    pos.append(self.walkers.pos.to('cpu').clone())
                idecor += 1

        if with_tqdm:
            log.info(
                "   Acceptance rate     : {:1.2f} %", (rate / self.nstep * 100))
            log.info(
                "   Timing statistics   : {:1.2f} steps/sec.", self.nstep / (time() - tstart))
            log.info(
                "   Total Time          : {:1.2f} sec.", (time() - tstart))

        return torch.cat(pos).requires_grad_()
