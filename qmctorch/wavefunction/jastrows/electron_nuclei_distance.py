import torch
from torch import nn


class ElectronNucleiDistance(nn.Module):

    def __init__(self, nelec, atomic_pos, ndim=3, scale=0.6):
        """Computes the electron-nuclei distances

        Args:
            nelec (int): number of electrons
            atomic_pos (tensor): positions of the atoms
            ndim (int): number of spatial dimensions
            scale(float, optional): value of the scale factor, Defaults to 0.6

        Examples::
            >>> endist = ElectronNucleiDistance(2,2)
            >>> epos = torch.tensor(500,6)
            >>> r = edist(pos)
            >>> dr = edist(pos,derivative=1)

        """

        super(ElectronNucleiDistance, self).__init__()
        self.nelec = nelec
        self.atoms = atomic_pos
        self.ndim = ndim
        self.kappa = scale

    def forward(self, inp, derivative=0):
        """Compute the pairwise distance between the electrons
        or its derivative. \n

        When required, the derivative is computed wrt to the first electron i.e.

        .. math::
            \\frac{dr_{ij}}{dx_i}

        which is different from :

        .. math::
            \\frac{d r_{ij}}{dx_j}

        Args:
            input (torch.tesnor): position of the electron \n
                                  size : Nbatch x [Nelec x Ndim]
            derivative (int, optional): degre of the derivative. \n
                                        Defaults to 0.

        Returns:
            torch.tensor: distance (or derivative) matrix \n
                          Nbatch x Nelec x Nelec if derivative = 0 \n
                          Nbatch x Ndim x  Nelec x Nelec if derivative = 1,2

        """

        # get the distance matrices
        input_ = input.view(-1, self.nelec, self.ndim)
        dist = self._get_distance_quadratic(input_)

        if derivative == 0:
            return dist

        elif derivative == 1:

            invr = (1. / dist).unsqueeze(1)
            diff_axis = input_.transpose(1, 2).unsqueeze(3)
            diff_axis = diff_axis - diff_axis.transpose(2, 3)
            return diff_axis * invr

        elif derivative == 2:

            invr3 = (1. / (dist**3)).unsqueeze(1)
            diff_axis = input_.transpose(1, 2).unsqueeze(3)
            diff_axis = (diff_axis - diff_axis.transpose(2, 3))**2

            diff_axis = diff_axis[:, [
                [1, 2], [2, 0], [0, 1]], ...].sum(2)
            return (diff_axis * invr3)

    @staticmethod
    def _get_distance_quadratic(input):
        """Compute the distance following a quadratic expansion

        Arguments:
            input {torch.tensor} -- electron position [nbatch x nelec x ndim]

        Returns:
            torch.tensor -- distance matrices nbatch x nelec x ndim]
        """

        norm = (input**2).sum(-1).unsqueeze(-1)
        norm_atom = (self.atoms)**2.unsqueeze(-1).T
        dist = (norm + norm_atom - 2.0 * input@self.atoms.T)
        return dist
