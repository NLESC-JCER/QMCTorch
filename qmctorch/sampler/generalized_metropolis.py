from tqdm import tqdm
from typing import Dict, Any, Callable, Optional
import torch
from torch.autograd import Variable, grad
from torch.distributions import MultivariateNormal
import numpy as np

from .sampler_base import SamplerBase
from .. import log


class GeneralizedMetropolis(SamplerBase):
    def __init__(  # pylint: disable=dangerous-default-value
        self,
        nwalkers: int = 100,
        nstep: int = 1000,
        step_size: float = 3,
        ntherm: int = -1,
        ndecor: int = 1,
        nelec: int = 1,
        ndim: int = 1,
        init: Dict[str, Any] = {"type": "uniform", "min": -5, "max": 5},
        cuda: bool = False,
    ) -> None:
        """Generalized Metropolis Hasting sampler

        Args:
            nwalkers (int, optional): number of walkers. Defaults to 100.
            nstep (int, optional): number of steps. Defaults to 1000.
            step_size (float, optional): size of the steps. Defaults to 3.
            ntherm (int, optional): number of steps for thermalization. Defaults to -1.
            ndecor (int, optional): number of steps for decorelation. Defaults to 1.
            nelec (int, optional): number of electron. Defaults to 1.
            ndim (int, optional): number of dimensions. Defaults to 1.
            init (dict, optional): method to initialize the walkers. Defaults to {'type': 'uniform', 'min': -5, 'max': 5}.
            cuda (bool, optional): use cuda. Defaults to False.

        Returns:
            None
        """

        SamplerBase.__init__(
            self, nwalkers, nstep, step_size, ntherm, ndecor, nelec, ndim, init, cuda
        )

    def __call__(
        self,
        pdf: Callable[[torch.Tensor], torch.Tensor],
        pos: Optional[torch.Tensor] = None,
        with_tqdm: bool = True,
    ) -> torch.Tensor:
        """Generate a series of point using MC sampling

        Args:
            pdf (callable): probability distribution function to be sampled
            pos (torch.Tensor, optional): position to start with.
                                          Defaults to None.
            with_tqdm (bool, optional): use tqdm to monitor progress

        Returns:
            torch.Tensor: positions of the walkers
        """
        with torch.no_grad():
            if self.ntherm < 0:
                self.ntherm = self.nstep + self.ntherm

            self.walkers.initialize(pos=pos)

            xi = self.walkers.pos.clone()
            xi.requires_grad = True

            rhoi = pdf(xi)
            drifti = self.get_drift(pdf, xi)

            rhoi[rhoi == 0] = 1e-16
            pos, rate, idecor = [], 0, 0

            rng = tqdm(
                range(self.nstep),
                desc="INFO:QMCTorch|  Sampling",
                disable=not with_tqdm,
            )

            for istep in rng:
                # new positions
                xf = self.move(drifti)

                # new function
                rhof = pdf(xf)
                driftf = self.get_drift(pdf, xf)
                rhof[rhof == 0.0] = 1e-16

                # transtions
                Tif = self.trans(xi, xf, driftf)
                Tfi = self.trans(xf, xi, drifti)
                pmat = (Tif * rhof) / (Tfi * rhoi).double()

                # accept the moves
                index = self._accept(pmat)

                # acceptance rate
                rate += index.byte().sum().float() / self.walkers.nwalkers

                # update position/function value
                xi[index, :] = xf[index, :]
                rhoi[index] = rhof[index]
                rhoi[rhoi == 0] = 1e-16

                drifti[index, :] = driftf[index, :]

                if istep >= self.ntherm:
                    if idecor % self.ndecor == 0:
                        pos.append(xi.clone().detach())
                    idecor += 1

            log.options(style="percent").debug(
                "  Acceptance rate %1.3f" % (rate / self.nstep * 100)
            )

            self.walkers.pos.data = xi.data

        return torch.cat(pos).requires_grad_()

    def move(self, drift: torch.Tensor) -> torch.Tensor:
        """Move electron one at a time in a vectorized way.

        Args:
            drift (torch.Tensor): drift velocity of the walkers

        Returns:
            torch.Tensor: new positions of the walkers
        """

        # Clone and reshape data to (nwalkers, nelec, ndim)
        new_pos = self.walkers.pos.clone()
        new_pos = new_pos.view(self.walkers.nwalkers, self.nelec, self.ndim)

        # Get random indices for electrons to move
        index = torch.LongTensor(self.walkers.nwalkers).random_(0, self.nelec)

        # Update positions of selected electrons
        new_pos[range(self.walkers.nwalkers), index, :] += self._move(drift, index)

        # Return reshaped positions
        return new_pos.view(self.walkers.nwalkers, self.nelec * self.ndim)

    def _move(self, drift: torch.Tensor, index: int) -> torch.Tensor:
        """Move a walker.

        Args:
            drift (torch.Tensor): drift velocity
            index (int): index of the electron to move

        Returns:
            torch.Tensor: position of the walkers
        """

        # Reshape drift to (nwalkers, nelec, ndim)
        d = drift.view(self.walkers.nwalkers, self.nelec, self.ndim)

        # Create a multivariate normal distribution with mean 0 and variance step_size
        mv = MultivariateNormal(
            torch.zeros(self.ndim), np.sqrt(self.step_size) * torch.eye(self.ndim)
        )

        # Add the drift to the random normal variable
        return (
            self.step_size * d[range(self.walkers.nwalkers), index, :]
            + mv.sample((self.walkers.nwalkers, 1)).squeeze()
        )

    def trans(
        self, xf: torch.Tensor, xi: torch.Tensor, drifti: torch.Tensor
    ) -> torch.Tensor:
        """Transform the positions

        Args:
            xf (torch.Tensor): Final positions
            xi (torch.Tensor): Initial positions
            drifti (torch.Tensor): Drift velocity

        Returns:
            torch.Tensor: Transition probabilities
        """
        a = (xf - xi - drifti * self.step_size).norm(dim=1)
        return torch.exp(-0.5 * a / self.step_size)

    def get_drift(
        self, pdf: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        """Compute the drift velocity

        Args:
            pdf (callable): function that returns the density
            x (torch.tensor): positions of the walkers

        Returns:
            torch.tensor: drift velocity
        """
        with torch.enable_grad():
            x.requires_grad = True
            rho = pdf(x).view(-1, 1)
            z = Variable(torch.ones_like(rho))
            grad_rho = grad(rho, x, grad_outputs=z, only_inputs=True)[0]
            return 0.5 * grad_rho / rho

    def _accept(self, P: torch.Tensor) -> torch.Tensor:
        """Accept the move or not

        Args:
            P (torch.Tensor): probability of each move

        Returns:
            torch.Tensor: the index of the accepted moves
        """
        P[P > 1] = 1.0
        tau = torch.rand(self.walkers.nwalkers, dtype=torch.float64)
        index = (P - tau >= 0).reshape(-1)
        return index.type(torch.bool)
