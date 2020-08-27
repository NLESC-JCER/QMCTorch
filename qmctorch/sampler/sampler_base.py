import torch
from .walkers import Walkers
from .. import log


class SamplerBase(object):

    def __init__(self, mol, nwalkers, nsample, nstep, step_size,
                 ntherm, ndecor, init, cuda):
        """Base class for the sampler

        Args:
            nwalkers (int): number of walkers
            nstep (int): number of MC steps
            step_size (float): size of the steps in bohr
            ntherm (int): number of MC steps to thermalize
            ndecor (int): unmber of MC steps to decorellate
            nelec (int): number of electrons in the system
            ndim (int): number of cartesian dimension
            init (dict): method to initialize the walkers
            cuda ([type]): [description]
        """

        self.nwalkers = nwalkers
        self.nsample = nsample
        self.nelec = mol.nelec
        self.ndim = 3
        self.nstep = nstep
        self.step_size = step_size
        self.ntherm = ntherm
        self.ndecor = ndecor
        self.cuda = cuda

        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.walkers = Walkers(
            nwalkers=nwalkers, nelec=nelec, ndim=self.ndim, init=mol.domain[init], cuda=cuda)

        # configure the sampler
        self.configure()

        # precision
        _type_ = torch.get_default_dtype()
        if _type_ == torch.float32:
            self.epsilon = 1E-7
        elif _type_ == torch.float64:
            self.epsilon = 1E-16

        self.log_base_data()

    def log_base_data(self):
        """log data about sampler."""
        log.info('')
        log.info(' Monte-Carlo Sampler')
        log.info('  Number of walkers   : {0}', self.nwalkers)
        log.info('  Thermalization steps: {0}', self.ntherm)
        if self.last_only:
            log.info('  Last position only')
        else:
            log.info('  Total number of step: {0}', self.nstep)
            log.info('  Decorelation steps  : {0}', self.ndecor)
            log.info('  Number of samples   : {0}', self.nsample)
        log.info('  Step size           : {0}', self.step_size)

    def __call__(self, pdf, *args, **kwargs):
        raise NotImplementedError(
            "Sampler must have a __call__ method")

    def __repr__(self):
        return self.__class__.__name__ + ' sampler with  %d walkers' % self.nwalkers

    def configure(self):
        """Configure the sampling

        Raises:
            ValueError: if nstep and nsample are both specified
        """
        if (self.nsample is not None) and (self.nstep is not None):
            raise ValueError(
                'nsample and nstep were specified at the same time')

        # get the number of step if we specified number of sample
        # i.e. each walkers collect several point along the traj
        if self.nsample is not None:
            self.last_only = False
            self.nstep = self.get_number_steps()

        # set the options if we specified the number of step
        # i.e. each walkers collect the last point only
        else:
            self.last_only = True
            self.nstep = self.ntherm
            self.nsample = self.walkers.nwalkers
            self.ndecor = 1

    def get_number_steps(self):
        """Define the number of steps to be performed."""
        if self.ndecor is None:
            raise ValueError(
                'You must specify a decorelation time via ndecor')
        if self.nsample < self.nwalkers:
            raise ValueError('nsample is lower than nwalkers')
        ninterval = self.nsample // self.nwalkers - 1
        nstep = self.ntherm + self.ndecor * ninterval
        return nstep

    def get_sampling_size(self):
        """evaluate the number of sampling point we'll have."""
        if self.ntherm == -1:
            return self.nwalkers
        else:
            return self.walkers.nwalkers * int((self.nstep-self.ntherm)/self.ndecor)
