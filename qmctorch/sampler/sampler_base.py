import torch
from .walkers import Walkers


class SamplerBase(object):

    def __init__(self, nwalkers, nstep, step_size, ntherm, ndecor, nelec, ndim, init, with_tqdm):
        """Base class for the sampler.

        Arguments:
            nwalkers {int]} -- number of walkers
            nstep {[int]} -- number of MC step
            step_size {[float]} -- size of the MC step
            nelec {int} -- number of electrons
            ndim {int} -- number of dimension per elec
            init {dict} -- method/data to initialize th walkers
        """

        self.nwalkers = nwalkers
        self.nelec = nelec
        self.ndim = ndim
        self.nstep = nstep
        self.step_size = step_size
        self.ntherm = ntherm
        self.ndecor = ndecor
        self.cuda = False
        self.device = torch.device('cpu')
        self.with_tqdm = with_tqdm

        self.walkers = Walkers(
            nwalkers=nwalkers, nelec=nelec, ndim=ndim, init=init)

    def __call__(self, pdf, *args, **kwargs):
        raise NotImplementedError(
            "Sampler must have a __call__ method")
