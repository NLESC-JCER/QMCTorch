import math
from time import time
from typing import Callable, Union

import torch
from torch.distributions import MultivariateNormal
from torch.nn import Parameter
from torch.optim import AdamW
from tqdm import tqdm

from .sampler_base import SamplerBase
from .utils.multimodal import MultiModalDistribution
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
                 nstacked: int = 1,
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
        self.mol = mol
        init = mol.domain('atomic')

        torch.nn.Module.__init__(self)
        SamplerBase.__init__(self, nwalkers, nstep,
                             step_size, ntherm, ndecor,
                             nelec, ndim, init, cuda)

        sigmas = []
        for i in range(1, nstacked + 1):
            for ne in init['atom_nelec']:
                sigmas.append(torch.eye(self.ndim, dtype=torch.double) * (ne / i) + (
                        torch.randn(self.ndim, self.ndim).abs() * 0.01))

        sigma = torch.stack(sigmas) / sum(init['atom_nelec'])
        weights = torch.tensor(init['atom_nelec'] * nstacked, dtype=torch.double) / (sum(init['atom_nelec']) * nstacked)

        # print(init['atom_coords'])
        self.mean = Parameter(data=torch.tensor(init['atom_coords'], dtype=torch.double), requires_grad=True)
        # self.sigma = Parameter(data=sigma, requires_grad=True)
        self.sigma = Parameter(data=torch.ones(self.nelec, self.ndim, dtype=torch.double), requires_grad=True)
        self.weights = Parameter(data=weights, requires_grad=True)

        # print([p for p in self.parameters()])

        self.natom = mol.natom
        self.nstacked = nstacked

        self.k = int(self.nwalkers * 0.01)

        self.logspace = logspace
        self.log_data()

    def log_data(self):
        """log data about the sampler."""
        pass

    def proposal(self):
        dists = []
        for i in range(self.natom * self.nstacked):
            # print(self.sigma[i].diag())
            dists.append(MultivariateNormal(self.mean[i], self.sigma[i].diag()))
        return MultiModalDistribution(dists, self.weights)

    def proposal_pdf(self, proposal, samples):
        return proposal.pdf(samples.view(self.nwalkers * self.nelec, self.ndim)).view(self.nwalkers, -1).sum(1)

    def sample(self, distribution):
        return distribution.sample(self.nwalkers * self.nelec).view(self.nwalkers, self.nelec * self.ndim)

    def accept(self, pdf, proposal, new_pos):
        f = pdf(new_pos).squeeze()
        g = self.proposal_pdf(proposal, new_pos)

        m = -torch.kthvalue(-f / g, self.k).values  # get kth largest value
        # m = (f / g).max()
        # print("max:", m)

        u = torch.rand_like(g)
        gum = g * u * m

        # accept the moves
        accepted = gum.lt(f)

        return accepted, f, g, m

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
        else:
            eps = 1E-4

        if self.ntherm >= self.nstep:
            raise ValueError('Thermalisation longer than trajectory')
        elif self.ntherm < 0:
            self.ntherm = self.nstep + self.ntherm

        opt = AdamW(self.parameters(), lr=self.step_size)

        pos, rate, idecor = [], 0, 0

        rng = tqdm(range(self.nstep),
                   desc='INFO:QMCTorch|  Sampling',
                   disable=not with_tqdm)
        tstart = time()
        self.walkers.initialize()

        for istep in rng:
            opt.zero_grad()  # zero the gradient buffers

            proposal = self.proposal()

            new_pos = self.sample(proposal)

            accepted, f, g, m = self.accept(pdf, proposal, new_pos)

            self.walkers.pos[accepted] = new_pos[accepted]

            # acceptance rate
            rate += torch.sum(accepted) / self.nwalkers

            # loss = (f * (f * m / g).log()).sum()
            loss = (g * m * (g * m / f).log()).sum()
            if istep % 100 == 0:
                # print("-------------")
                # print("mean:", self.mean.data)
                # print("sigma:", self.sigma.data)
                print("acc rate:", (torch.sum(accepted) / self.nwalkers).item())
                print("loss:", loss.item())
            loss.backward()
            opt.step()
            self.sigma.data = torch.clamp(self.sigma, min=eps)
            self.weights.data = torch.clamp(self.weights, min=eps)

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
