
import torch

class DensityVarianceKernel(object):
    def __init__(self, atomic_pos, sigma=1.0, scale_factor=1.0):
        self.atomic_pos = atomic_pos.unsqueeze(0).unsqueeze(1)
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.nelec = None
        self.ndim = None

    def __call__(self, x):
        d = self.get_estimate_density(x)
        out = self.sigma * (1.0 - d).sum(-1)
        return out.unsqueeze(-1)

    def get_atomic_distance(self, pos):
        nwalkers = pos.shape[0]
        pos = pos.view(nwalkers, self.nelec, self.ndim)
        dist = pos.unsqueeze(-2) - self.atomic_pos
        return dist.norm(dim=-1)

    def get_estimate_density(self, pos):
        d = self.get_atomic_distance(pos)
        d = torch.exp(-self.scale_factor * d**2)
        return d


class CenterVarianceKernel(object):
    def __init__(self, sigma=1.0, scale_factor=1.0):
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.nelec = None
        self.ndim = None

    def __call__(self, x):
        d = self.get_estimate_density(x)
        out = self.sigma * (1.0 - d)
        return out.unsqueeze(-1)

    def get_estimate_density(self, pos):
        nwalkers = pos.shape[0]
        pos = pos.view(nwalkers, self.nelec, self.ndim)
        d = pos.norm(dim=-1)
        d = torch.exp(-self.scale_factor * d**2)
        return d


class ConstantVarianceKernel(object):
    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, x):
        return self.sigma
