from tqdm import tqdm
import torch
from torch.distributions import MultivariateNormal
from time import time
from typing import Callable, Union, Dict
from .sampler_base import SamplerBase
from .. import log


class StateDependentNormalProposal(object):

    def __init__(self, kernel, nelec, ndim, device):

        self.ndim = ndim
        self.nelec = nelec
        self.kernel = kernel
        self.device = device
        self.multiVariate = MultivariateNormal(
            torch.zeros(self.ndim), 1. * torch.eye(self.ndim))

    def __call__(self, x):
        nwalkers = x.shape[0]
        scale = self.kernel(x)
        displacement = self.multiVariate.sample(
            (nwalkers, self.nelec)).to(self.device)
        displacement *= scale
        return displacement.view(nwalkers, self.nelec*self.ndim)

    def get_transition_ratio(self, x, y):
        sigmax = self.kernel(x)
        sigmay = self.kernel(y)

        rdist = (x-y).view(-1, self.nelec,
                           self.ndim).norm(dim=-1).unsqueeze(-1)

        prefac = (sigmax/sigmay)**(self.ndim/2)
        tratio = torch.exp(-0.5*rdist**2 *
                           (1./sigmay-1./sigmax))
        tratio *= prefac

        return tratio.squeeze().prod(-1)
