import numpy as np
import torch
from torch.distributions import MultivariateNormal
from typing import Union, Dict
from .. import log


class Walkers:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        nwalkers: int = 100,
        nelec: int = 1,
        ndim: int = 3,
        init: Union[Dict, None] = None,
        cuda: bool = False,
    ) -> None:
        """Creates Walkers for the sampler.

        Args:
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            nelec (int, optional): number of electron. Defaults to 1.
            ndim (int, optional): Number of dimensions. Defaults to 3.
            init (dict, optional): method to initialize the walkers. Defaults to None. (see Molecule.domain())
            cuda (bool, optional): turn cuda ON/OFF. Defaults to False
        """
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.nelec = nelec
        self.init_domain = init

        self.pos = None
        self.status = None

        self.cuda = cuda

        if cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def initialize(self, pos: Union[None, torch.Tensor] = None) -> None:
        """Initalize the position of the walkers

        Args:
            method (str, optional): how to initialize the positions. Defaults to 'uniform'.
            pos ([type], optional): existing position of the walkers. Defaults to None.

        Raises:
            ValueError: if the method is not recognized
        """
        if self.cuda:
            self.device = torch.device("cuda")

        if pos is not None:
            if len(pos) > self.nwalkers:
                pos = pos[-self.nwalkers :, :]
            self.pos = pos

        else:
            log.debug("  Initialize walkers")
            if "center" in self.init_domain.keys():
                self.pos = self._init_center()

            elif "min" in self.init_domain.keys():
                self.pos = self._init_uniform()

            elif "mean" in self.init_domain.keys():
                self.pos = self._init_multivar()

            elif "atom_coords" in self.init_domain.keys():
                self.pos = self._init_atomic()

            else:
                raise ValueError("Init walkers not recognized")

    def _init_center(self) -> torch.Tensor:
        """Initialize the walkers at the center of the molecule

        Returns:
            torch.tensor: positions of the walkers, shape (nwalkers, nelec * ndim)
        """
        eps = 1e-3
        pos = -eps + 2 * eps * torch.rand(self.nwalkers, self.nelec * self.ndim)
        return pos.type(torch.get_default_dtype()).to(device=self.device)

    def _init_uniform(self) -> torch.Tensor:
        """Initialize the walkers in a box covering the molecule

        Returns:
            torch.tensor: positions of the walkers, shape (nwalkers, nelec * ndim)
        """
        pos = torch.rand(self.nwalkers, self.nelec * self.ndim)
        pos *= self.init_domain["max"] - self.init_domain["min"]
        pos += self.init_domain["min"]
        return pos.type(torch.get_default_dtype()).to(device=self.device)

    def _init_multivar(self) -> torch.Tensor:
        """Initialize the walkers in a sphere covering the molecule.

        Returns:
            torch.Tensor: positions of the walkers, shape (nwalkers, nelec * ndim)
        """
        # Create a multivariate normal distribution with the given mean and covariance
        multi = MultivariateNormal(
            torch.as_tensor(self.init_domain["mean"]),
            torch.as_tensor(self.init_domain["sigma"]),
        )
        # Sample positions for the walkers and cast to the default dtype
        pos = multi.sample((self.nwalkers, self.nelec)).type(torch.get_default_dtype())
        # Reshape the sampled positions to match the expected output shape
        pos = pos.view(self.nwalkers, self.nelec * self.ndim)
        # Move the positions to the appropriate device (CPU or GPU)
        return pos.to(device=self.device)

    def _init_atomic(self) -> torch.Tensor:
        """Initialize the walkers around the atoms.

        Positions are distributed around atomic coordinates with some randomness.

        Returns:
            torch.Tensor: Positions of the walkers, shape (nwalkers, nelec * ndim).
        """
        pos = torch.zeros(self.nwalkers, self.nelec * self.ndim)
        idx_ref, nelec_tot = [], 0

        nelec_placed, natom = [], 0
        for iat, nelec in enumerate(self.init_domain["atom_nelec"]):
            idx_ref += [iat] * nelec
            nelec_tot += nelec
            natom += 1

        for iw in range(self.nwalkers):
            nelec_placed = [0] * natom
            idx = torch.as_tensor(idx_ref)
            idx = idx[torch.randperm(nelec_tot)]
            xyz = torch.as_tensor(self.init_domain["atom_coords"])[idx, :]

            for ielec in range(nelec_tot):
                _idx = idx[ielec]
                if nelec_placed[_idx] == 0:
                    s = 1.0 / self.init_domain["atom_num"][_idx]
                elif nelec_placed[_idx] < 5:
                    s = 2.0 / (self.init_domain["atom_num"][_idx] - 2)
                else:
                    s = 3.0 / (self.init_domain["atom_num"][_idx] - 3)
                xyz[ielec, :] += np.random.normal(scale=s, size=(1, 3))
                nelec_placed[_idx] += 1

            pos[iw, :] = xyz.view(-1)
        return pos.to(device=self.device)
