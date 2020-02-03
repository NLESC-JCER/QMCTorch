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

        self.cuda = False
        self.device = torch.device('cpu')

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
        if self.cuda:
            self.device = torch.device('cuda')

        if pos is not None:
            if len(pos) > self.nwalkers:
                pos = pos[-self.nwalkers:, :]
            self.pos = pos

        else:

            if 'center' in self.init_domain.keys():
                self.pos = self._init_center()

            elif 'min' in self.init_domain.keys():
                self.pos = self._init_uniform()

            elif 'mean' in self.init_domain.keys():
                self.pos = self._init_multivar()

            else:
                raise ValueError('Init walkers not recognized')

    def _init_center(self):
        eps = 1E-6
        pos = -eps + 2*eps*torch.rand(self.nwalkers, self.nelec*self.ndim)
        return pos.type(torch.get_default_dtype()).to(device=self.device)

    def _init_uniform(self):
        pos = torch.rand(self.nwalkers, self.nelec*self.ndim)
        pos *= (self.init_domain['max'] - self.init_domain['min'])
        pos += self.init_domain['min']
        return pos.type(torch.get_default_dtype()).to(device=self.device)

    def _init_multivar(self):
        multi = MultivariateNormal(
            torch.tensor(self.init_domain['mean']),
            torch.tensor(self.init_domain['sigma']))
        pos = multi.sample((self.nwalkers, self.nelec)).type(
            torch.get_default_dtype())
        pos = pos.view(self.nwalkers, self.nelec*self.ndim)
        return pos.to(device=self.device)
