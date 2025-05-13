import torch
from torch import nn
from typing import Tuple, Union
from .scaling import (
    get_scaled_distance,
    get_der_scaled_distance,
    get_second_der_scaled_distance,
)


class ElectronNucleiDistance(nn.Module):
    def __init__(
        self,
        nelec: int,
        atomic_pos: torch.Tensor,
        ndim: int = 3,
        scale: bool = False,
        scale_factor: float = 0.6,
    ) -> None:
        """Computes the electron-nuclei distances

        .. math::
            r_{iA} = \\sqrt{ (x_i-x_A)^2 + (y_i-y_A)^2 + (z_i-z_A)^2}

        Args:
            nelec (int): number of electrons
            atomic_pos (torch.tensor): positions of the atoms
            ndim (int): number of spatial dimensions
            scale(bool, optional): return scaled values, Defaults to False
            scale_factor(float, optional): value of the scale factor,
                                           Defaults to 0.6

        Examples::
            >>> endist = ElectronNucleiDistance(2,2)
            >>> epos = torch.tensor(500,6)
            >>> r = edist(pos)
            >>> dr = edist(pos,derivative=1)

        """
        super().__init__()
        self.nelec: int = nelec
        self.atoms: torch.Tensor = atomic_pos
        self.ndim: int = ndim
        self.scale: bool = scale
        self.kappa: float = scale_factor

        _type_ = torch.get_default_dtype()
        if _type_ == torch.float32:
            self.eps = 1e-6
        elif _type_ == torch.float64:
            self.eps = 1e-16

    def forward(
        self, input: torch.Tensor, derivative: int = 0
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the pairwise distances between electrons and atoms
        or their derivative.

        Args:
            input (torch.Tensor): position of the electron \n
                                  size : Nbatch x [Nelec x Ndim]
            derivative (int, optional): degre of the derivative. \n
                                        Defaults to 0.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                distance (or derivative) matrix \n
                Nbatch x Nelec x Natom if derivative = 0 \n
                Nbatch x Ndim x  Nelec x Natom if derivative = 1,2

        """

        # get the distance matrices
        input_ = input.view(-1, self.nelec, self.ndim)
        dist = self._get_distance_quadratic(input_, self.atoms)
        dist = torch.sqrt(dist)

        if derivative == 0:  # pylint: disable=no-else-return
            if self.scale:
                return get_scaled_distance(self.kappa, dist)
            else:
                return dist

        elif derivative == 1:
            der_dist = self.get_der_distance(input_, dist)
            if self.scale:
                return get_der_scaled_distance(self.kappa, dist, der_dist)
            else:
                return der_dist

        elif derivative == 2:
            d2_dist = self.get_second_der_distance(input_, dist)

            if self.scale:
                der_dist = self.get_der_distance(input_, dist)
                return get_second_der_scaled_distance(
                    self.kappa, dist, der_dist, d2_dist
                )
            else:
                return d2_dist

    def get_der_distance(self, pos: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """Get the derivative of the electron-nuclei distance matrix

        .. math::
            \\frac{d r_{iA}}{d x_i}


        Args:
            pos (torch.tensor): positions of the electrons
                                Nbatch x Nelec x Ndim
            dist (torch.tensor): distance matrix between the elecs
                           Nbatch x Nelec x Nelec

        Returns:
            [type]: [description]
        """
        invr = (1.0 / (dist + self.eps)).unsqueeze(-1)
        diff_axis = (pos.unsqueeze(-1) - self.atoms.T).transpose(2, 3)
        return (diff_axis * invr).permute(0, 3, 1, 2)

    def get_second_der_distance(
        self, pos: torch.Tensor, dist: torch.Tensor
    ) -> torch.Tensor:
        """Get the derivative of the electron-nuclei distance matrix

        .. math::
            \\frac{d^2 r_{iA}}{d x_i^2}

        Args:
            pos (torch.tensor): positions of the electrons
                                Nbatch x Nelec x Ndim
            dist (torch.tensor): distance matrix between the elecs
                           Nbatch x Nelec x Nelec

        Returns:
            [type]: [description]
        """
        invr3 = (1.0 / (dist**3 + self.eps)).unsqueeze(1)
        diff_axis = pos.transpose(1, 2).unsqueeze(3)
        diff_axis = (diff_axis - self.atoms.T.unsqueeze(1)) ** 2

        diff_axis = diff_axis[:, [[1, 2], [2, 0], [0, 1]], ...].sum(2)

        return diff_axis * invr3

    @staticmethod
    def _get_distance_quadratic(
        elec_pos: torch.Tensor, atom_pos: torch.Tensor
    ) -> torch.Tensor:
        """Compute the distance following a quadratic expansion

        Arguments:
            input {torch.tensor} -- electron position [nbatch x nelec x ndim]

        Returns:
            torch.tensor -- distance matrices nbatch x nelec x ndim]
        """
        norm = (elec_pos**2).sum(-1).unsqueeze(-1)
        norm_atom = (atom_pos**2).sum(-1).unsqueeze(-1).T
        dist = norm + norm_atom - 2.0 * elec_pos @ atom_pos.T
        return dist
