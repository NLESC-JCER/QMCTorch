import torch
from typing import Callable
from torch.distributions import MultivariateNormal


class StateDependentNormalProposal(object):
    def __init__(
        self,
        kernel: Callable[[torch.Tensor], torch.Tensor],
        nelec: int,
        ndim: int,
        device: torch.device,
    ) -> None:
        """
        Initialize StateDependentNormalProposal.

        Args:
            kernel: A callable that takes a tensor of shape (nwalkers, nelec*ndim)
                and returns a tensor of shape (nwalkers, nelec*ndim).
            nelec: The number of electrons.
            ndim: The number of dimensions.
            device: The device to use for computations.
        """
        self.ndim = ndim
        self.nelec = nelec
        self.kernel = kernel
        self.device = device
        self.multiVariate = MultivariateNormal(
            torch.zeros(self.ndim), 1.0 * torch.eye(self.ndim)
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the proposal distribution

        Args:
            x: The current position of the walkers, shape (nwalkers, nelec*ndim)

        Returns:
            The displacement, shape (nwalkers, nelec*ndim)
        """
        nwalkers = x.shape[0]
        scale = self.kernel(x)  # shape (nwalkers, nelec*ndim)
        displacement = self.multiVariate.sample((nwalkers, self.nelec))  # shape (nwalkers, nelec, ndim)
        displacement *= scale  # shape (nwalkers, nelec, ndim)
        return displacement.view(nwalkers, self.nelec * self.ndim)

    def get_transition_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the transition ratio for the Metropolis-Hastings acceptance probability.

        Args:
            x: The current position of the walkers, shape (nwalkers, nelec*ndim)
            y: The proposed position of the walkers, shape (nwalkers, nelec*ndim)

        Returns:
            The transition ratio, shape (nwalkers,)
        """
        sigmax = self.kernel(x)
        sigmay = self.kernel(y)

        rdist = (x - y).view(-1, self.nelec, self.ndim).norm(dim=-1).unsqueeze(-1)

        prefac = (sigmax / sigmay) ** (self.ndim / 2)
        tratio = torch.exp(-0.5 * rdist**2 * (1.0 / sigmay - 1.0 / sigmax))
        tratio *= prefac

        return tratio.squeeze().prod(-1)
