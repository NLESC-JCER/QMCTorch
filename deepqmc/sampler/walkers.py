import torch
from torch.distributions import MultivariateNormal
import numpy as np


class Walkers(object):

    def __init__(self, nwalkers=100, nelec=1, ndim=1, init=None, move=None):

        self.nwalkers = nwalkers
        self.ndim = ndim
        self.nelec = nelec
        self.init_domain = self._get_init_domain(init)
        self.move = move

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
            domain['sigma'] = np.diag(np.std(mol.atom_coords, 0)+0.5)
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

    def move(self, step_size):
        """Main swith to propose new moves

        Args:
            step_size (float): size of the MC moves
            method (str, optional): move electrons one at a time or all at the
                                    same time.
                                    'one' or 'all' . Defaults to 'one'.

        Returns:
            torch.tensor: new positions of the walkers
        """

        if self.nelec == 1:
            new_pos = self._move_only_elec(step_size)

        else:
            new_pos = self._move_one_vect(step_size)

        return new_pos

    def _move_only_elec(self, step_size):
        """Move the only electron there.

        Args:
            step_size (float): size of the MC moves

        Returns:
            torch.tensor: new positions of the walkers
        """
        _size = (self.nwalkers, self.nelec*self.ndim)
        return self.pos + self._get_new_coord(step_size, _size)

    def _move_one_vect(self, step_size):
        """Move electron one at a time in a vectorized way.

        Args:
            step_size (float): size of the MC moves

        Returns:
            torch.tensor: new positions of the walkers
        """

        # clone and reshape data : Nwlaker, Nelec, Ndim
        new_pos = self.pos.clone()
        new_pos = new_pos.view(self.nwalkers, self.nelec, self.ndim)

        # get indexes
        index = torch.LongTensor(self.nwalkers).random_(0, self.nelec)

        # change selected data
        new_pos[range(self.nwalkers), index,
                :] += self._get_new_coord(step_size, (self.nwalkers, self.ndim))

        return new_pos.view(self.nwalkers, self.nelec*self.ndim)

    def _get_new_coord(self, step_size, size):

        if self.move == 'uniform':
            return self._get_uniform(step_size, size)

        elif self.move == 'drift':
            raise ValueError('drift sampling not implemtented yet')

    @staticmethod
    def _get_uniform(step_size, size):
        """Return a random array of length size between
        [-step_size,step_size]

        Args:
            step_size (float): boundary of the array
            size (int): number of points in the array

        Returns:
            torch.tensor: random array
        """
        return step_size * (2 * torch.rand(size) - 1)
