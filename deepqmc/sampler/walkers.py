import torch
from torch.distributions import MultivariateNormal


class Walkers(object):

    def __init__(self, nwalkers=100, nelec=1, ndim=1, init=None):

        self.nwalkers = nwalkers
        self.ndim = ndim
        self.nelec = nelec
        self.init_domain = init

        self.pos = None
        self.status = None

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

            if self.init_domain is None:
                self.pos = torch.zeros((self.nwalkers, self.nelec*self.ndim))

            elif 'min' in self.init_domain.keys():
                self.pos = self._init_uniform()

            elif 'mean' in self.init_domain.keys():
                self.pos = self._init_multivar()

            else:
                raise ValueError('Init walkers not recognized')

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
