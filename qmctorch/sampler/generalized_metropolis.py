from tqdm import tqdm
import torch
from torch.autograd import Variable, grad
from torch.distributions import MultivariateNormal
import numpy as np

from .sampler_base import SamplerBase
from .metropolis import Metropolis
from .. import log


class GeneralizedMetropolis(Metropolis):

    def __init__(self, nwalkers=100,
                 nstep=1000, step_size=0.2,
                 ntherm=-1, ndecor=1,
                 nelec=1, ndim=3,
                 init={'min': -5, 'max': 5},
                 move={'type': 'all-elec', 'proba': 'normal'},
                 cuda=False):
        """Generalized Metropolis Hasting generator

        Args:
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            nstep (int, optional): Number of steps. Defaults to 1000.
            step_size (int, optional): length of the step. Defaults to 3.
            nelec (int, optional): total number of electrons. Defaults to 1.
            ntherm (int, optional): number of mc step to thermalize. Defaults to -1, i.e. keep ponly last position
            ndecor (int, optional): number of mc step for decorelation. Defauts to 1.
            ndim (int, optional): total number of dimension. Defaults to 1.
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
            >>> wf = Orbital(mol)
            >>> sampler = GeneralizedMetropolis(nwalkers=100, nelec=wf.nelec)
            >>> pos = sampler(wf.pdf)
        """

        SamplerBase.__init__(self, nwalkers, nstep,
                             step_size, ntherm, ndecor,
                             nelec, ndim, init, move, cuda)

        if self.movedict['type'] != 'all-elec':
            raise ValueError(
                'Generalized Metropolis only implemented for all elecron moves')

    def transisition_matrix(self):
        """computes the transitions matrix"""

        Tif = self.trans(self.walkers.pos,
                         self.data.final_pos, self.data.final_drift)
        Tfi = self.trans(self.data.final_pos,
                         self.walkers.pos, self.data.initial_drift)

        return (Tif * self.data.final_density) / \
            (Tfi * self.data.initial_density).double()

    def init_sampling_data(self, pdf):
        """Computes the data needed to stat the sampling."""
        super(Metropolis, self).init_sampling_data(pdf)

        xi = self.walkers.pos.clone()
        xi.requires_grad = True

        self.data.initial_drift = self.get_drift(pdf, xi)
        self.data.final_drift = None

    def update_sampling_data(self, index):
        """Update the data for th sampling process

        Args:
            index (torch tensor): indices of the accepted move
        """
        super().update_sampling_data(index)
        self.data.initial_drift[index,
                                :] = self.data.final_drift[index, :]

    def propose_move(self, pdf, id_elec):
        """propose a new move and computes the data

        Args:
            pdf (callable): density
            id_elec (torch.tensor): indexes of the elecs to move
        """
        super().propose_move(pdf, id_elec)
        self.final_drift = self.get_drift(pdf, self.data.final_pos)

    def move(self, id_elec):
        """Move electron one at a time in a vectorized way.

        Args:


        Returns:
            torch.tensor: new positions of the walkers
        """

        if self.nelec == 1 or self.movedict['type'] = 'all-elec':
            return self._move(self.walkers.pos, self.nelec)

        # clone and reshape data : Nwlaker, Nelec, Ndim
        new_pos = self.walkers.pos.clone()
        new_pos = new_pos.view(self.nwalkers,
                               self.nelec, self.ndim)

        # get indexes
        if id_elec is None:
            index = torch.LongTensor(self.nwalkers).random_(
                0, self.nelec)
        else:
            index = torch.LongTensor(self.nwalkers).fill_(id_elec)

        # change selected data
        new_pos[range(self.nwalkers), index,
                :] += self._move(1)

        # new_pos[range(self.nwalkers), index,
        #         :] += self._move(self.data.initial_drift, index)

        return new_pos.view(self.nwalkers, self.nelec * self.ndim)

    def _move(self, initial_pos, num_elec, index=None):

        d = self.data.initial_drift.view(self.nwalkers,
                                         self.nelec, self.ndim)

        mv = MultivariateNormal(torch.zeros(self.ndim), np.sqrt(
            self.step_size) * torch.eye(self.ndim))

        displacement self.step_size * d[range(self.nwalkers), index, :] \
            + mv.sample((self.nwalkers, 1)).squeeze()

    def trans(self, xf, xi, drifti):
        a = (xf - xi - drifti * self.step_size).norm(dim=1)
        return torch.exp(- 0.5 * a / self.step_size)

    def get_drift(self, pdf, x):
        return 0.5 * pdf(xi, return_grad=True) / pdf(xi)
