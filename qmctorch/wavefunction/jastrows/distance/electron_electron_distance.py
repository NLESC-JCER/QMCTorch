import torch
from torch import nn
from .scaling import (get_scaled_distance,
                      get_der_scaled_distance,
                      get_second_der_scaled_distance)


class ElectronElectronDistance(nn.Module):

    def __init__(self, nelec, ndim=3, scale=False, scale_factor=0.6):
        """Computes the electron-electron distances

        .. math::
            r_{ij} = \\sqrt{ (x_i-x_j)^2 + (y_i-y_j)^2 + (z_i-z_j)^2}

        Args:
            nelec (int): number of electrons
            ndim (int): number of spatial dimensions
            scale(bool, optional): return scaled values, Defaults to False
            scale_factor(float, optional): value of the scale factor,
                                           Defaults to 0.6

        Examples::
            >>> edist = ElectronDistance(2,3)
            >>> pos = torch.tensor(500,6)
            >>> r = edist(pos)
            >>> dr = edist(pos,derivative=1)

        """

        super().__init__()
        self.nelec = nelec
        self.ndim = ndim
        self.scale = scale
        self.kappa = scale_factor

        _type_ = torch.get_default_dtype()
        if _type_ == torch.float32:
            self.eps = 1E-6
        elif _type_ == torch.float64:
            self.eps = 1E-16

    def forward(self, input, derivative=0):
        """Compute the pairwise distance between the electrons
        or its derivative. \n

        When required, the derivative is computed wrt to the first electron i.e.

        .. math::
            \\frac{dr_{ij}}{dx_i}

        which is different from :

        .. math::
            \\frac{d r_{ij}}{dx_j} = -\\frac{dr_{ij}}{dx_i}

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
        dist = self.get_distance_quadratic(input_)
        dist = self.safe_sqrt(dist)

        if derivative == 0:
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
                return get_second_der_scaled_distance(self.kappa,
                                                      dist,
                                                      der_dist,
                                                      d2_dist)
            else:
                return d2_dist

    def safe_sqrt(self, dist):
        """Compute the square root of the electron electron distance matrix.

        Args:
            dist (torch.tensor): ee distances squared
                                 Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: sqrt of dist Nbatch x Nelec x Nelec

        """

        # epsilon on the diag needed for back prop
        eps_ = self.eps * \
            torch.diag(dist.new_ones(dist.shape[-1])).expand_as(dist)

        # extact the diagonal as diag can be negative someties
        # due to numerical noise
        diag = torch.diag_embed(
            torch.diagonal(
                dist, dim1=-1, dim2=-2))

        # remove diagonal and add eps for backprop
        dist = torch.sqrt(dist - diag + eps_)

        return dist

    def get_der_distance(self, pos, dist):
        """Get the derivative of the electron electron distance matrix.

        .. math::
            \\frac{d r_{ij}}{d x_i}

        Args:
            pos (torch.tensor): positions of the electrons
                                Nbatch x Nelec x Ndim
            dist (torch.tensor): distance matrix between the elecs
                           Nbatch x Nelec x Nelec

        Returns:
            [type]: [description]
        """

        eps_ = self.eps * \
            torch.diag(dist.new_ones(
                dist.shape[-1])).expand_as(dist)

        invr = (1. / (dist + eps_)).unsqueeze(1)
        diff_axis = pos.transpose(1, 2).unsqueeze(3)
        diff_axis = diff_axis - diff_axis.transpose(2, 3)
        return diff_axis * invr

    def get_second_der_distance(self, pos, dist):
        """Get the second derivative of the electron electron distance matrix.

        .. math::
            \\frac{d^2 r_{ij}}{d x_i^2}

        Args:
            pos (torch.tensor): positions of the electrons
                                Nbatch x Nelec x Ndim
            dist (torch.tensor): distance matrix between the elecs
                           Nbatch x Nelec x Nelec

        Returns:
            [type]: [description]
        """

        eps_ = self.eps * \
            torch.diag(dist.new_ones(
                dist.shape[-1])).expand_as(dist)
        invr3 = (1. / (dist**3 + eps_)).unsqueeze(1)
        diff_axis = pos.transpose(1, 2).unsqueeze(3)
        diff_axis = (diff_axis - diff_axis.transpose(2, 3))**2

        diff_axis = diff_axis[:, [
            [1, 2], [2, 0], [0, 1]], ...].sum(2)
        return (diff_axis * invr3)

    @staticmethod
    def get_distance_quadratic(pos):
        """Compute the distance following a quadratic expansion

        Arguments:
            pos {torch.tensor} -- electron position [nbatch x nelec x ndim]

        Returns:
            torch.tensor -- distance matrices nbatch x nelec x ndim]
        """

        norm = (pos**2).sum(-1).unsqueeze(-1)
        dist = (norm + norm.transpose(1, 2) - 2.0 *
                torch.bmm(pos, pos.transpose(1, 2)))
        return dist

    @staticmethod
    def get_difference(pos):
        """Compute the difference ri - rj

        Arguments:
            pos {torch.tensor} -- electron position [nbatch x nelec x ndim]

        Returns:
            torch.tensor -- distance matrices nbatch x nelec x nelec x ndim]
        """
        out = pos[:, :, None, :] - pos[:, None, :, :]
        return out
