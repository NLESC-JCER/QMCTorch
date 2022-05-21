import torch
from typing import List


class MultiModalDistribution:
    def __init__(self, distributions: List[torch.distributions.Distribution], weights: torch.Tensor):
        self.n = len(distributions)
        self.distributions = distributions
        self.weights = weights / weights.sum()

    def sample(self, size):
        if isinstance(size, List):
            size = size[0]

        sample_sizes = torch.bincount(self.weights.multinomial(size, replacement=True))

        samples = torch.Tensor()
        for d, s in zip(self.distributions, sample_sizes):
            samples = torch.cat((samples, d.sample([s])))
        return samples

    def pdf(self, samples: torch.Tensor):
        if len(samples.shape) == 1:
            samples = torch.unsqueeze(samples, -1)

        ret = torch.zeros(samples.shape[:-1])
        for d, w in zip(self.distributions, self.weights):
            lp = d.log_prob(samples)
            if lp.shape != ret.shape:
                lp = lp.sum(-1)
            ret += lp.exp() * w

        return ret

    def log_prob(self, samples: torch.Tensor):
        ret = torch.zeros((self.n,) + samples.shape)
        for i, (d, w) in enumerate(zip(self.distributions, self.weights)):
            ret[i] = d.log_prob(samples) + w.log()
        return torch.logsumexp(ret, 0)


if __name__ == "__main__":
    from torch.distributions import Normal, Uniform
    import matplotlib.pyplot as plt

    n1 = Normal(0, 1)
    # n2 = Normal(4, 1)
    n2 = Uniform(0, 1, validate_args=False)

    _weights = torch.tensor([4., 1.])

    mm = MultiModalDistribution([n1, n2], _weights)

    _samples = mm.sample(100000)

    plt.hist(_samples.numpy(), bins=50, density=True)
    sorted_samples = torch.sort(_samples).values
    plt.plot(sorted_samples, mm.pdf(sorted_samples))
    plt.show()
