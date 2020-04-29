import torch
import numpy as np
from torch.distributions import MultivariateNormal


class Walkers(object):

    def __init__(self, nwalkers=100, nelec=1, ndim=1, init=None):
        """Creates Walkers for the sampler.

        Args:
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            nelec (int, optional): number of electron. Defaults to 1.
            ndim (int, optional): Number of dimensions. Defaults to 1.
            init (dict, optional): method to initialize the walkers. Defaults to None. (see Molecule.domain())
        """
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
            method (str, optional): how to initialize the positions. Defaults to 'uniform'.
            pos ([type], optional): existing position of the walkers. Defaults to None.

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

            elif 'atom_coords' in self.init_domain.keys():
                self.pos = self._init_atomic()

            else:
                raise ValueError('Init walkers not recognized')

    def _init_center(self):
        """Initialize the walkers at the center of the molecule

        Returns:
            torch.tensor: positions of the walkers
        """
        eps = 1E-3
        pos = -eps + 2 * eps * \
            torch.rand(self.nwalkers, self.nelec * self.ndim)
        return pos.type(
            torch.get_default_dtype()).to(
            device=self.device)

    def _init_uniform(self):
        """Initialize the walkers in a box covering the molecule

        Returns:
            torch.tensor: positions of the walkers
        """
        pos = torch.rand(self.nwalkers, self.nelec * self.ndim)
        pos *= (self.init_domain['max'] - self.init_domain['min'])
        pos += self.init_domain['min']
        return pos.type(
            torch.get_default_dtype()).to(
            device=self.device)

    def _init_multivar(self):
        """Initialize the walkers in a sphere covering the molecule

        Returns:
            torch.tensor -- positions of the walkers
        """
        multi = MultivariateNormal(
            torch.tensor(self.init_domain['mean']),
            torch.tensor(self.init_domain['sigma']))
        pos = multi.sample((self.nwalkers, self.nelec)).type(
            torch.get_default_dtype())
        pos = pos.view(self.nwalkers, self.nelec * self.ndim)
        return pos.to(device=self.device)

    def _init_atomic(self):
        """Initialize the walkers around the atoms

        Returns:
            torch.tensor -- positions of the walkers
        """
        pos = torch.zeros(self.nwalkers, self.nelec * self.ndim)
        idx_ref, nelec_tot = [], 0

        nelec_placed, natom = [], 0
        for iat, nelec in enumerate(self.init_domain['atom_nelec']):
            idx_ref += [iat] * nelec
            nelec_tot += nelec
            natom += 1

        for iw in range(self.nwalkers):

            nelec_placed = [0] * natom
            idx = torch.tensor(idx_ref)
            idx = idx[torch.randperm(nelec_tot)]
            xyz = torch.tensor(
                self.init_domain['atom_coords'])[
                idx, :]

            for ielec in range(nelec_tot):
                _idx = idx[ielec]
                if nelec_placed[_idx] == 0:
                    s = 1. / self.init_domain['atom_num'][_idx]
                elif nelec_placed[_idx] < 5:
                    s = 2. / (self.init_domain['atom_num'][_idx] - 2)
                else:
                    s = 3. / (self.init_domain['atom_num'][_idx] - 3)
                xyz[ielec,
                    :] += np.random.normal(scale=s, size=(1, 3))
                nelec_placed[_idx] += 1

            pos[iw, :] = xyz.view(-1)
        return pos
