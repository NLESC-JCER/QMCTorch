class SamplerBase(object):

    def __init__(self, walkers, nstep=1000,
                 step_size=3):

        self.walkers = walkers
        self.nwalkers = walkers.nwalkers
        self.nelec = walkers.nelec
        self.ndim = walkers.ndim
        self.nstep = nstep
        self.step_size = step_size

    def generate(self, pdf):
        raise NotImplementedError()
