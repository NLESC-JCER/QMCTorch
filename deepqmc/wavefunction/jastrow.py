import torch
from torch import nn


class ElectronDistance(nn.Module):

    def __init__(self, nelec, ndim):
        super(ElectronDistance, self).__init__()
        self.nelec = nelec
        self.ndim = ndim

    def forward(self, input, derivative=0):
        """compute the pairwise distance between two sets of electrons
        Or the derivative of these elements

        Args:
            input ([type]): [description]
            derivative (int, optional): degre of the derivative.
                                        Defaults to 0.

        Returns:
            torch.tensor: distance (or derivative) matrix
                          Nbatch x Nelec x Nelec if derivative = 0
                          Nbatch x Ndim x  Nelec x Nelec if derivative = 1,2

        """
        '''compute the pairwise distance between two sets of electrons.
        Args:
            input1 (Nbatch,Nelec1*Ndim) : position of the electrons
            input2 (Nbatch,Nelec2*Ndim) : position of the electrons
                                          if None -> input1
        Returns:
            mat (Nbatch,Nelec1,Nelec2) : pairwise distance between electrons
        '''

        input = input.view(-1, self.nelec, self.ndim)
        norm = (input**2).sum(-1).unsqueeze(-1)
        dist = norm + norm.transpose(1, 2) - 2.0 * \
            torch.bmm(input, input.transpose(1, 2))

        if derivative == 0:
            return dist

        elif derivative == 1:

            invr = (1./dist).unsqueeze_(1)
            diff_axis = input.transpose(1, 2).unsqueeze_(3)
            diff_axis = diff_axis - diff_axis.transpose(2, 3)
            return diff_axis * invr

        elif derivative == 2:

            invr3 = (1./dist**3).unsqueeze(1)
            diff_axis = input.transpose(1, 2).unsqueeze_(3)
            diff_axis = (diff_axis - diff_axis.transpose(2, 3))**2
            diff_axis = diff_axis[:, [[1, 2], [2, 0], [0, 1]], ...].sum(2)
            return diff_axis * invr3


class TwoBodyJastrowFactor(nn.Module):

    def __init__(self, nup, ndown):
        super(TwoBodyJastrowFactor, self).__init__()

        self.nup = nup
        self.ndown = ndown
        self.nelec = nup+ndown
        self.ndim = 3

        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.weight.requires_grad = True

        bup = torch.cat((0.25*torch.ones(nup, nup), 0.5 *
                         torch.ones(nup, ndown)), dim=1)

        bdown = torch.cat((0.5*torch.ones(ndown, nup), 0.25 *
                           torch.ones(ndown, ndown)), dim=1)
        self.static_weight = torch.cat((bup, bdown), dim=0)

        self.edist = ElectronDistance(self.nelec, self.ndim)

    def forward(self, pos, derivative=0):
        """Compute the Jastrow factors as :

        .. math::
            J(ri,rj) = \Prod_{i,j} \exp(B_{ij}) with
            B_{ij} = \frac{b r_{i,j}}{1+b'r_{i,j}}

        Args:
            pos ([type]): [description]
            derivative (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """

        r = self.edist(pos)
        jast = self._get_jastrow_elements(r)

        if derivative == 0:
            return self._unique_pair_prod(jast)

        elif derivative == 1:
            return self._jastrow_derivative(r, jast)

    def _jastrow_derivative(self, r, jast):

        nbatch = r.shape[0]
        dr = self.edist(pos, derivative=1)
        djast = self._get_der_jastrow_elements(r, dr) * jast.unsqueeze(1)

        grad_jast = torch.ones(nbatch, self.nelec, self.ndim)
        for i in range(self.nelec):

            for j in range(i+1, self.nelec):
                grad_jast[:, i] += self._unique_pair_prod(
                    jast, not_el=(i, j)) * djast[..., i, j]

            for j in range(0, i):
                grad_jast[:, i] += self._unique_pair_prod(
                    jast, not_el=(j, i)) * djast[..., j, i]

        return grad_jast

    def _get_jastrow_elements(self, r):
        """Get the elements of the jastrow matrix :
        .. math::
            out_{i,j} = \exp{ \frac{b r_{i,j}}{1+b'r_{i,j}} }

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the jastrow elements
                          Nbatch x Nelec x Nelec
        """
        return torch.exp(self.static_weight * r /
                         (1.0 + self.weight * r))

    def _get_der_jastrow_elements(self, r, dr):
        """Get the elements of the derivative of the jastrow matrix
        .. math::

            out_{k,i,j} = A + B
            A_{kij} = b \frac{dr_{ij}}{dk_i} / (1+b'r_{ij})
            B_{kij} = - b b' r_{ij} \frac{dr_{ij}}{dk_i} / (1+b'r_{ij})^ 2

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the derivative of the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """

        r.unsqueeze_(1)
        denom = 1. / (1.0 + self.weight * r)

        a = self.static_weight * dr * denom
        b = - self.static_weight * self.weight * r * dr * denom**2
        r.squeeze_()
        return (a+b)

    def _get_second_der_jastrow_elements(self, r, dr, d2r):
        """Get the elements of the pure 2nd derivative of the jastrow matrix

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec
            d2r (torch.tensor): matrix of the 2nd derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the pure 2nd derivative of the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """

        r.unsqueeze_(1)
        denom = 1. / (1.0 + self.weight * r)

        a = self.static_weight * d2r * denom
        b = -2 * self.static_weight * self.weight * dr * denom**2
        c = -2 * self.static_weight * self.weight**2 * r * dr * denom**3
        r.squeeze_()
        return a+b+c

    def _unique_pair_prod(self, mat, not_el=None):
        """Compute the product of the lower mat elements

        Args:
            mat (torch.tensor): input matrix [..., N x N]
            not_el (tuple(i,j), optional):
                single element to exclude of the product.
                Defaults to None.

        Returns:
            torch.tensor : value of the product
        """

        mat_cpy = mat.clone()
        if not_el is not None:
            i, j = not_el
            mat_cpy[..., i, j] = 1

        return mat_cpy[:, torch.tril(
            torch.ones(self.nelec, self.nelec)) == 0].prod(1).view(-1, 1)


if __name__ == "__main__":

    pos = torch.rand(10, 12)
    jastrow = TwoBodyJastrowFactor(2, 2)
    val = jastrow(pos)

    r = jastrow.edist(pos)
    dr = jastrow.edist(pos, derivative=1)
    d2r = jastrow.edist(pos, derivative=2)
    dval = jastrow(pos, derivative=1)
