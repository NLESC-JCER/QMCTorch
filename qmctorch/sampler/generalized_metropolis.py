from tqdm import tqdm
import torch
from torch.autograd import Variable, grad
from torch.distributions import MultivariateNormal
import numpy as np

from .sampler_base import SamplerBase
from .. import log


class GeneralizedMetropolis(SamplerBase):

    def __init__(self, nwalkers=100, nstep=1000, step_size=3,
                 ntherm=-1, ndecor=1,
                 nelec=1, ndim=1,
                 init={'type': 'uniform', 'min': -5, 'max': 5},
                 cuda=False):
        """Generalized Metropolis Hasting sampler

        Args:
            nwalkers (int, optional): number of walkers. Defaults to 100.
            nstep (int, optional): number of steps. Defaults to 1000.
            step_size (int, optional): size of the steps. Defaults to 3.
            ntherm (int, optional): number of steps for thermalization. Defaults to -1.
            ndecor (int, optional): number of steps for decorelation. Defaults to 1.
            nelec (int, optional): number of electron. Defaults to 1.
            ndim (int, optional): number of dimensions. Defaults to 1.
            init (dict, optional): method to initialize the walkers. Defaults to {'type': 'uniform', 'min': -5, 'max': 5}.
            cuda (bool, optional): use cuda. Defaults to False.
        """

        SamplerBase.__init__(self, nwalkers, nstep,
                             step_size, ntherm, ndecor, nelec, ndim, init,
                             cuda)

    def __call__(self, pdf, pos=None, with_tqdm=True):
        """Generate a series of point using MC sampling

        Args:
            pdf (callable): probability distribution function to be sampled
            pos (torch.tensor, optional): position to start with.
                                          Defaults to None.
            with_tqdm (bool, optional): use tqdm to monitor progress

        Returns:
            torch.tensor: positions of the walkers
        """
        with torch.no_grad():

            if self.ntherm < 0:
                self.ntherm = self.nstep + self.ntherm

            self.walkers.initialize(pos=pos)

            xi = self.walkers.pos.clone()
            xi.requires_grad = True

            rhoi = pdf(xi)
            drifti = self.get_drift(pdf, xi)

            rhoi[rhoi == 0] = 1E-16
            pos, rate, idecor = [], 0, 0

            rng = tqdm(range(self.nstep),
                       desc='INFO:QMCTorch|  Sampling',
                       disable=not with_tqdm)

            for istep in rng:

                # new positions
                xf = self.move(drifti)

                # new function
                rhof = pdf(xf)
                driftf = self.get_drift(pdf, xf)
                rhof[rhof == 0.] = 1E-16

                # transtions
                Tif = self.trans(xi, xf, driftf)
                Tfi = self.trans(xf, xi, drifti)
                pmat = (Tif * rhof) / (Tfi * rhoi).double()

                # accept the moves
                index = self._accept(pmat)

                # acceptance rate
                rate += index.byte().sum().float() / self.nwalkers

                # update position/function value
                xi[index, :] = xf[index, :]
                rhoi[index] = rhof[index]
                rhoi[rhoi == 0] = 1E-16

                drifti[index, :] = driftf[index, :]

                if (istep >= self.ntherm):
                    if (idecor % self.ndecor == 0):
                        pos.append(xi.clone().detach())
                    idecor += 1

            log.options(style='percent').debug("  Acceptance rate %1.3f" %
                                               (rate / self.nstep * 100))

            self.walkers.pos.data = xi.data

        return torch.cat(pos).requires_grad_()

    def move(self, drift):
        """Move electron one at a time in a vectorized way.

        Args:
            drift (torch.tensor): drift velocity of the walkers

        Returns:
            torch.tensor: new positions of the walkers
        """

        # clone and reshape data : Nwlaker, Nelec, Ndim
        new_pos = self.walkers.pos.clone()
        new_pos = new_pos.view(self.nwalkers,
                               self.nelec, self.ndim)

        # get indexes
        index = torch.LongTensor(self.nwalkers).random_(
            0, self.nelec)

        new_pos[range(self.nwalkers), index,
                :] += self._move(drift, index)

        return new_pos.view(self.nwalkers, self.nelec * self.ndim)

    def _move(self, drift, index):
        """Move a walker.

        Args:
            drift (torch.tensor): drift velocity
            index (int): indx of the electron to move

        Returns:
            torch.tensor: position of the walkers
        """

        d = drift.view(self.nwalkers,
                       self.nelec, self.ndim)

        mv = MultivariateNormal(torch.zeros(self.ndim), np.sqrt(
            self.step_size) * torch.eye(self.ndim))

        return self.step_size * d[range(self.nwalkers), index, :] \
            + mv.sample((self.nwalkers, 1)).squeeze()

    def trans(self, xf, xi, drifti):
        """transform the positions

        Args:
            xf ([type]): [description]
            xi ([type]): [description]
            drifti ([type]): [description]

        Returns:
            [type]: [description]
        """
        a = (xf - xi - drifti * self.step_size).norm(dim=1)
        return torch.exp(- 0.5 * a / self.step_size)

    def get_drift(self, pdf, x):
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
            grad_rho = grad(rho, x,
                            grad_outputs=z,
                            only_inputs=True)[0]
            return 0.5 * grad_rho / rho

    def _accept(self, P):
        """accept the move or not

        Args:
            P (torch.tensor): probability of each move

        Returns:
            torch.tensor: the indx of the accepted moves
        """
        P[P > 1] = 1.0
        tau = torch.rand(self.walkers.nwalkers).double()
        index = (P - tau >= 0).reshape(-1)
        return index.type(torch.bool)
