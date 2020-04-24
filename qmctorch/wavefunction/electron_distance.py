import torch
from torch import nn


class ElectronDistance(nn.Module):

    def __init__(self, nelec, ndim):
        """Computes the electron-electron distances

        Args:
            nelec (int): number of electrons
            ndim (int): number of spatial dimensions 

        Examples::
            >>> edist = ElectronDistance(2,3)
            >>> pos = torch.tensor(500,6)
            >>> r = edist(pos)
            >>> dr = edist(pos,derivative=1)

        """

        super(ElectronDistance, self).__init__()
        self.nelec = nelec
        self.ndim = ndim

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

        # eosilon on the diag needed for back prop
        eps_ = self.eps * \
            torch.diag(dist.new_ones(dist.shape[-1])).expand_as(dist)

        # extact the diagonal as diag can be negative someties
        # due to numerical noise
        diag = torch.diag_embed(
            torch.diagonal(
                dist, dim1=-1, dim2=-2))

        # remove diagonal and add eps for backprop
        dist = torch.sqrt(dist - diag + eps_)

        if derivative == 0:
            return dist

        elif derivative == 1:

            eps_ = self.eps * \
                torch.diag(dist.new_ones(
                    dist.shape[-1])).expand_as(dist)

            invr = (1. / (dist + eps_)).unsqueeze(1)
            diff_axis = input_.transpose(1, 2).unsqueeze(3)
            diff_axis = diff_axis - diff_axis.transpose(2, 3)
            return diff_axis * invr

        elif derivative == 2:

            eps_ = self.eps * \
                torch.diag(dist.new_ones(
                    dist.shape[-1])).expand_as(dist)
            invr3 = (1. / (dist**3 + eps_)).unsqueeze(1)
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
        dist = (norm + norm.transpose(1, 2) - 2.0 *
                torch.bmm(input, input.transpose(1, 2)))
        return dist

    def _get_distance_diff(self, input):
        """Compute the distance following a difference scheme

        Arguments:
            input {torch.tensor} -- electron position [nbatch x nelec x ndim]

        Returns:
            torch.tensor -- distance matrices nbatch x nelec x ndim]
        """
        nbatch = input.shape[0]
        in1 = input.unsqueeze(1).expand(
            nbatch, self.nelec, self.nelec, self.ndim)
        in2 = input.unsqueeze(2).expand(
            nbatch, self.nelec, self.nelec, self.ndim)
        dist = torch.pow(in1 - in2, 2).sum(3)
        return dist
