import torch
from .walkers import Walkers


class SamplerBase(object):

    def __init__(self, nwalkers, nstep, step_size, nelec, ndim, init, move):
        """Base class for the sampler.

        Arguments:
            nwalkers {int]} -- number of walkers
            nstep {[int]} -- number of MC step
            step_size {[float]} -- size of the MC step
            nelec {int} -- number of electrons
            ndim {int} -- number of dimension per elec
            init {dict} -- method/data to initialize th walkers
            move {dict} -- method/data to perform the move
        """

        self.nwalkers = nwalkers
        self.nelec = nelec
        self.ndim = ndim
        self.nstep = nstep
        self.step_size = step_size
        self.movedict = move
        self.cuda = False
        self.device = torch.device('cpu')

        self.walkers = Walkers(
            nwalkers=nwalkers, nelec=nelec, ndim=ndim, init=init)

    def generate(self, pdf):
        raise NotImplementedError()
