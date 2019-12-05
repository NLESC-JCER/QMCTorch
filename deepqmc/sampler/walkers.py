import torch

class Walkers(object):

    def __init__(self,nwalkers,nelec,ndim,domain):

        self.nwalkers = nwalkers
        self.ndim = ndim
        self.nelec = nelec
        self.domain = domain

        self.pos = None
        self.status = None

    def initialize(self, method='uniform', pos=None):

        if pos is not None:
            if len(pos) > self.nwalkers:
                pos = pos[-self.nwalkers:,:]
            self.pos = pos

        else:
            options = ['center','uniform']
            if method not in options:
                raise ValueError('method %s not recognized. Options are : %s ' %(method, ' '.join(options)) )

            if method == options[0]:
                self.pos = torch.zeros((self.nwalkers, self.nelec*self.ndim ))

            elif method == options[1]:
                self.pos = torch.rand(self.nwalkers, self.nelec*self.ndim)
                self.pos *= (self.domain['max'] - self.domain['min'])
                self.pos += self.domain['min']

        self.status = torch.ones((self.nwalkers,1))

    def move(self, step_size, method='one'):

        if method == 'one':
            new_pos = self._move_one_vect(step_size)

        elif method == 'all':
            new_pos = self._move_all(step_size)

        return new_pos

    def _move_all(self,step_size):
        return self.pos + self.status * self._random(step_size,(self.nwalkers,self.nelec * self.ndim))

    def _move_one_vect(self,step_size):

        # clone and reshape data : Nwlaker, Nelec, Ndim
        new_pos = self.pos.clone()
        new_pos = new_pos.view(self.nwalkers,self.nelec,self.ndim)

        # get indexes
        index = torch.LongTensor(self.nwalkers).random_(0,self.nelec)

        # change selected data
        new_pos[range(self.nwalkers),index,:] += self._random(step_size,(self.nwalkers,self.ndim))

        return new_pos.view(self.nwalkers,self.nelec*self.ndim)

    def _random(self,step_size,size):
        return step_size * (2 * torch.rand(size) - 1)
