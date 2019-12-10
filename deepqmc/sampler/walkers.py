import torch
from torch.distributions import MultivariateNormal
import numpy as np


class Walkers(object):

    def __init__(self, nwalkers=100, nelec=1, ndim=1, init=None):

        self.nwalkers = nwalkers
        self.ndim = ndim
        self.nelec = nelec
        self.init_domain = self._get_init_domain(init)

        self.pos = None
        self.status = None

    @staticmethod
    def _get_init_domain(init):

        domain = dict()
        mol, method = init
        domain['type'] = method

        if method == 'uniform':
            domain['min'] = np.min(mol.atom_coords) - 0.5
            domain['max'] = np.max(mol.atom_coords) + 0.5

        if method == 'sphere':
            domain['mean'] = np.mean(mol.atom_coords, 0)
            domain['sigma'] = np.diag(np.std(mol.atom_coords, 0)+0.25)
        return domain

    def initialize(self, pos=None):
        """Initalize the position of the walkers

        Args:
            method (str, optional): how to initialize the positions.
                                    Defaults to 'uniform'.
            pos ([type], optional): existing position of the walkers.
                                    Defaults to None.

        Raises:
            ValueError: if the method is not recognized
        """

        if pos is not None:
            if len(pos) > self.nwalkers:
                pos = pos[-self.nwalkers:, :]
            self.pos = pos

        else:
            options = ['center', 'uniform', 'sphere']
            if self.init_domain['type'] not in options:
                raise ValueError('method %s not recognized. Options are : %s '
                                 % (self.init_domain['type'], ' '.join(options)))

            if self.init_domain['type'] == options[0]:
                self.pos = torch.zeros((self.nwalkers, self.nelec*self.ndim))

            elif self.init_domain['type'] == options[1]:
                self.pos = self._init_uniform()

            elif self.init_domain['type'] == options[2]:
                self.pos = self._init_multivar()

    def _init_uniform(self):
        pos = torch.rand(self.nwalkers, self.nelec*self.ndim)
        pos *= (self.init_domain['max'] - self.init_domain['min'])
        pos += self.init_domain['min']
        return pos

    def _init_multivar(self):
        multi = MultivariateNormal(
            torch.tensor(self.init_domain['mean']),
            torch.tensor(self.init_domain['sigma']))
        pos = multi.sample((self.nwalkers, self.nelec))
        return pos.view(self.nwalkers, self.nelec*self.ndim).float()
