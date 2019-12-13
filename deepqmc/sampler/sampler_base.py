from deepqmc.sampler.walkers import Walkers


class SamplerBase(object):

    def __init__(self, nwalkers, nstep, step_size, nelec, ndim, init):

        self.nwalkers = nwalkers
        self.nelec = nelec
        self.ndim = ndim
        self.nstep = nstep
        self.step_size = step_size

        self.walkers = Walkers(
            nwalkers=nwalkers, nelec=nelec, ndim=ndim, init=init)

    def generate(self, pdf):
        raise NotImplementedError()
