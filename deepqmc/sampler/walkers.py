import torch
from torch.distributions import MultivariateNormal


class Walkers(object):

    def __init__(self, nwalkers, nelec, ndim, domain):

        self.nwalkers = nwalkers
        self.ndim = ndim
        self.nelec = nelec
        self.domain = domain

        self.pos = None
        self.status = None

    def initialize(self, method='uniform', pos=None):
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
            if method not in options:
                raise ValueError('method %s not recognized. Options are : %s '
                                 % (method, ' '.join(options)))

            if method == options[0]:
                self.pos = torch.zeros((self.nwalkers, self.nelec*self.ndim))

            elif method == options[1]:
                self.pos = self._uniform()

            elif method == options[2]:
                self.pos = self._multivar()

        self.status = torch.ones((self.nwalkers, 1))

    def _uniform(self):
        pos = torch.rand(self.nwalkers, self.nelec*self.ndim)
        pos *= (self.domain['max'] - self.domain['min'])
        pos += self.domain['min']
        return pos

    def _multivar(self):
        multi = MultivariateNormal(
            torch.tensor(self.domain['mean']),
            torch.tensor(self.domain['sigma']))
        pos = multi.sample((self.nwalkers, self.nelec))
        return pos.view(self.nwalkers, self.nelec*self.ndim).float()

    def move(self, step_size, method='one'):
        """Main swith to propose new moves

        Args:
            step_size (float): size of the MC moves
            method (str, optional): move electrons one at a time or all at the
                                    same time.
                                    'one' or 'all' . Defaults to 'one'.

        Returns:
            torch.tensor: new positions of the walkers
        """

        if method == 'one':
            new_pos = self._move_one_vect(step_size)

        elif method == 'all':
            new_pos = self._move_all(step_size)

        return new_pos

    def _move_all(self, step_size):
        """Move all electrons at the same time.

        Args:
            step_size (float): size of the MC moves

        Returns:
            torch.tensor: new positions of the walkers
        """
        _size = (self.nwalkers, self.nelec*self.ndim)
        return self.pos + self.status * self._random(step_size, _size)

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
                :] += self._random(step_size, (self.nwalkers, self.ndim))

        return new_pos.view(self.nwalkers, self.nelec*self.ndim)

    @staticmethod
    def _random(step_size, size):
        """Return a random array of length size between
        [-step_size,step_size]

        Args:
            step_size (float): boundary of the array
            size (int): number of points in the array

        Returns:
            torch.tensor: random array
        """
        return step_size * (2 * torch.rand(size) - 1)
