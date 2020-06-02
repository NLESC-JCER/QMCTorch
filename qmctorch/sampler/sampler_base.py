import torch
from .walkers import Walkers


class SamplerBase(object):

    def __init__(self, nwalkers, nstep, step_size,
                 ntherm, ndecor, nelec, ndim, init,
                 cuda, with_tqdm):
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
            with_tqdm ([type]): [description]
        """

        self.nwalkers = nwalkers
        self.nelec = nelec
        self.ndim = ndim
        self.nstep = nstep
        self.step_size = step_size
        self.ntherm = ntherm
        self.ndecor = ndecor
        self.cuda = cuda
        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.with_tqdm = with_tqdm

        self.walkers = Walkers(
            nwalkers=nwalkers, nelec=nelec, ndim=ndim, init=init, cuda=cuda)

    def __call__(self, pdf, *args, **kwargs):
        raise NotImplementedError(
            "Sampler must have a __call__ method")
