from typing import ContextManager, Tuple
import torch
from torch import nn
from ..wavefunction import WaveFunction


class Loss(nn.Module):
    def __init__(
        self,
        wf: WaveFunction,
        method: str = "energy",
        clip: bool = False,
        clip_threshold: int = 5,
    ):
        """Defines the loss to use during the optimization

        Arguments:
            wf {Wavefunction} -- wave function object used

        Keyword Arguments:
            method {str} -- method to use  (default: {'energy'})
                            (energy, variance, weighted-energy,
                            weighted-variance)
            clip {bool} -- clip the values that are +/- % sigma away from
                           the mean (default: {False})
        """

        super(Loss, self).__init__()

        self.wf = wf
        self.method = method
        self.clip = clip

        # by default we use weights
        # that are needed if we do
        # not resample at every time step
        self.use_weight = True

        # number of +/- std for clipping
        # Excludes values + /- Nstd x std the mean of the eloc
        self.clip_num_std = clip_threshold

        # select loss function
        self.loss_fn = {"energy": torch.mean, "variance": torch.var}[method]

        # init values of the weights
        self.weight = {"psi": None, "psi0": None}

    def forward(
        self, pos: torch.Tensor, no_grad: bool = False, deactivate_weight: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the loss

        Args:
            pos (torch.Tensor): Positions of the walkers in that batch
            no_grad (bool, optional): Computes the gradient of the loss
                                      (default: {False})
            deactivate_weight (bool, optional): Deactivates the weight computation
                                               (default: {False})

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Value of the loss, local energies
        """

        # check if grads are requested
        with self.get_grad_mode(no_grad):
            # compute local eneergies
            local_energies = self.wf.local_energy(pos)

            # mask the energies if necessary
            mask = self.get_clipping_mask(local_energies)

            # sampling_weight
            weight = self.get_sampling_weights(pos, deactivate_weight)

            # compute the loss
            loss = self.loss_fn((weight * local_energies)[mask])

        return loss, local_energies

    @staticmethod
    def get_grad_mode(no_grad: bool) -> ContextManager:
        """Returns a context manager to enable or disable gradient computation.

        Args:
            no_grad (bool): Whether to disable gradient computation.

        Returns:
            typing.ContextManager: A context manager to disable or enable gradient computation.
        """
        return torch.no_grad() if no_grad else torch.enable_grad()

    def get_clipping_mask(self, local_energies: torch.Tensor) -> torch.Tensor:
        """Computes the clipping mask.

        Args:
            local_energies (torch.Tensor): Values of the local energies.

        Returns:
            torch.Tensor: A boolean tensor representing the clipping mask.
        """
        if self.clip:
            median = torch.median(local_energies)
            std = torch.std(local_energies)
            zscore = torch.abs((local_energies - median) / std)
            mask = zscore < self.clip_num_std
        else:
            mask = torch.ones_like(local_energies).type(torch.bool)

        return mask

    def get_sampling_weights(
        self, pos: torch.Tensor, deactivate_weight: bool
    ) -> torch.Tensor:
        """Get the weight needed when resampling is not
        done at every step

        Args:
            pos (torch.Tensor): Positions of the walkers
            deactivate_weight (bool): Deactivate the computation of the weight

        Returns:
            torch.Tensor: The weight to apply to the local energy
        """

        local_use_weight = self.use_weight * (not deactivate_weight)

        if local_use_weight:
            # computes the weights
            self.weight["psi"] = self.wf(pos)

            # if we just resampled store psi and all w=1
            if self.weight["psi0"] is None:
                self.weight["psi0"] = self.weight["psi"].detach().clone()
                w = torch.ones_like(self.weight["psi"])

            # otherwise compute ration of psi
            else:
                w = (self.weight["psi"] / self.weight["psi0"]) ** 2
                w /= w.sum()  # should we multiply by the number of elements ?

            return w

        else:
            return torch.tensor(1.0)
