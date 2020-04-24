import torch
from torch import nn
from .electron_distance import ElectronDistance
from ..utils import register_extra_attributes


class TwoBodyJastrowFactor(nn.Module):

    def __init__(self, nup, ndown, w=1., cuda=False):
        """Computes the Pade-Jastrow factor

        .. math::
            J = \prod_{i<j} \exp(B_{ij}) \quad \quad \\text{with} \quad \quad
            B_{ij} = \\frac{w_0 r_{i,j}}{1 + w r_{i,j}}

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            w (float, optional): Value of the variational parameter. Defaults to 1..
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super(TwoBodyJastrowFactor, self).__init__()

        self.nup = nup
        self.ndown = ndown
        self.nelec = nup + ndown
        self.ndim = 3

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

        self.weight = nn.Parameter(torch.tensor([w]))
        self.weight.requires_grad = True

        bup = torch.cat((0.25 * torch.ones(nup, nup), 0.5 *
                         torch.ones(nup, ndown)), dim=1)

        bdown = torch.cat((0.5 * torch.ones(ndown, nup), 0.25 *
                           torch.ones(ndown, ndown)), dim=1)

        self.static_weight = torch.cat(
            (bup, bdown), dim=0).to(
            self.device)

        self.edist = ElectronDistance(self.nelec, self.ndim)

        register_extra_attributes(self, ['weight'])

    def _to_device(self):
        """Export the non parameter variable to the device."""

        self.device = torch.device('cuda')
        self.to(self.device)
        attrs = ['static_weight']
        for at in attrs:
            self.__dict__[at] = self.__dict__[at].to(self.device)

    def forward(self, pos, derivative=0, jacobian=True):
        """Compute the Jastrow factors.

        Args:
            pos (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0,1,2,).
                            Defaults to 0.
            jacobian (bool, optional): Return the jacobian (i.e. the sum of
                                       the derivatives) or the individual
                                       terms. Defaults to True.
                                       False only for derivative=1

        Returns:
            torch.tensor: value of the jastrow parameter for all confs
        """
        if not jacobian:
            assert(derivative == 1)

        size = pos.shape
        assert size[1] == self.nelec * self.ndim
        r = self.edist(pos)
        jast = self._get_jastrow_elements(r)

        if derivative == 0:

            return self._prod_unique_pairs(jast)

        elif derivative == 1:
            dr = self.edist(pos, derivative=1)
            return self._jastrow_derivative(r, dr, jast, jacobian)

        elif derivative == 2:
            dr = self.edist(pos, derivative=1)
            d2r = self.edist(pos, derivative=2)
            return self._jastrow_second_derivative(r, dr, d2r, jast)

    def _jastrow_derivative(self, r, dr, jast, jacobian):
        """Compute the value of the derivative of the Jastrow factor

        Args:
            r (torch.tensor): ee distance matrix Nbatch x Nelec x Nelec
            jast (torch.tensor): values of the ajstrow elements
                                 Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: gradient of the jastrow factors
                          Nbatch x Nelec x Ndim
        """
        if jacobian:
            djast = self._get_der_jastrow_elements(r, dr).sum(1)
            prod_val = self._prod_unique_pairs(jast)

        else:
            djast = self._get_der_jastrow_elements(r, dr)
            prod_val = self._prod_unique_pairs(jast).unsqueeze(-1)

        return (self._sum_unique_pairs(djast, axis=-1) -
                self._sum_unique_pairs(djast, axis=-2)) * prod_val

    def _jastrow_second_derivative(self, r, dr, d2r, jast):
        """Compute the value of the pure 2nd derivative of the Jastrow factor

        Args:
            r (torch.tensor): ee distance matrix Nbatch x Nelec x Nelec
            jast (torch.tensor): values of the ajstrow elements
                                 Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: diagonal hessian of the jastrow factors
                          Nbatch x Nelec x Ndim
        """

        # pure second derivative terms
        prod_val = self._prod_unique_pairs(jast)
        d2jast = self._get_second_der_jastrow_elements(
            r, dr, d2r).sum(1)
        hess_jast = 0.5 * (self._sum_unique_pairs(d2jast, axis=-1)
                           + self._sum_unique_pairs(d2jast, axis=-2))

        # mixed terms
        djast = (self._get_der_jastrow_elements(r, dr))  # .sum(1)
        hess_jast += self._partial_derivative(
            djast, out_mat=hess_jast)

        return hess_jast * prod_val

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
        return torch.exp(self._compute_kernel(r))

    def _compute_kernel(self, r):
        """ Get the jastrow kernel.
        .. math::
            B_{ij} = \frac{b r_{i,j}}{1+b'r_{i,j}}

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: matrix of the jastrow kernels
                          Nbatch x Nelec x Nelec
        """
        return self.static_weight * r / (1.0 + self.weight * r)

    def _get_der_jastrow_elements(self, r, dr):
        """Get the elements of the derivative of the jastrow kernels
        wrt to the first electrons

        .. math::

            d B_{ij} / d k_i =  d B_{ij} / d k_j  = - d B_{ji} / d k_i

            out_{k,i,j} = A1 + A2
            A1_{kij} = w0 \frac{dr_{ij}}{dk_i} / (1 + w r_{ij})
            A2_{kij} = - w0 w' r_{ij} \frac{dr_{ij}}{dk_i} / (1 + w r_{ij})^2

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the derivative of the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """

        r_ = r.unsqueeze(1)
        denom = 1. / (1.0 + self.weight * r_)
        a = self.static_weight * dr * denom
        b = - self.static_weight * self.weight * r_ * dr * denom**2

        return (a + b)

    def _get_second_der_jastrow_elements(self, r, dr, d2r):
        """Get the elements of the pure 2nd derivative of the jastrow kernels
        wrt to the first electron

        .. math ::

            d^2 B_{ij} / d k_i^2 =  d^2 B_{ij} / d k_j^2 = d^2 B_{ji} / d k_i^2

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec
            d2r (torch.tensor): matrix of the 2nd derivative of 
                                the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the pure 2nd derivative of 
                          the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """

        r_ = r.unsqueeze(1)
        denom = 1. / (1.0 + self.weight * r_)
        denom2 = denom**2
        dr_square = dr**2
        a = self.static_weight * d2r * denom
        b = -2 * self.static_weight * self.weight * dr_square * denom2
        c = - self.static_weight * self.weight * r_ * d2r * denom2
        d = 2 * self.static_weight * self.weight**2 * r_ * dr_square * denom**3

        e = self._get_der_jastrow_elements(r, dr)

        return a + b + c + d + e**2

    def _partial_derivative(self, djast, out_mat=None):
        """Get the product of the mixed second deriative terms.

        .. math ::

            d B_{ij} / d x_i * d B_{kl} / d x_k

        Args:
            djast (torch.tensor): first derivative of the jastrow kernels
            out_mat (torch.tensor, optional): output matrix. Defaults to None.

        Returns:
            torch.tensor:
        """

        if out_mat is None:
            nbatch = djast.shape[0]
            out_mat = torch.zeros(nbatch, self.nelec)

        for idx in range(self.nelec):

            index_pairs = [(idx, j, 1) for j in range(
                idx + 1, self.nelec)] + [(j, idx, -1) for j in range(0, idx)]

            for p1 in range(len(index_pairs) - 1):
                i1, j1, w1 = index_pairs[p1]
                for p2 in range(p1 + 1, len(index_pairs)):
                    i2, j2, w2 = index_pairs[p2]

                    d1 = djast[..., i1, j1] * w1
                    d2 = djast[..., i2, j2] * w2

                    out_mat[..., idx] += (d1 * d2).sum(1)

        return out_mat

    def _prod_unique_pairs(self, mat, not_el=None):
        """Compute the product of the lower mat elements

        Args:
            mat (torch.tensor): input matrix [..., N x N]
            not_el (tuple(i,j), optional):
                single element(s) to exclude of the product.
                Defaults to None.

        Returns:
            torch.tensor : value of the product
        """

        mat_cpy = mat.clone()
        if not_el is not None:

            if not isinstance(not_el, list):
                not_el = [not_el]

            for _el in not_el:
                i, j = _el
                mat_cpy[..., i, j] = 1

        return mat_cpy[..., torch.tril(torch.ones(
            self.nelec, self.nelec)) == 0].prod(1).view(-1, 1)

    def _sum_unique_pairs(self, mat, axis=None):
        """Sum the unique pairs of the lower triangluar matrix

        Args:
            mat (torch.tensor): input matrix [..., N x N]
            axis (int, optional): index of the axis to sum. Defaults to None.

        Returns:
            torch.tensor:
        """

        mat_cpy = mat.clone()
        mat_cpy[..., torch.tril(torch.ones(
            self.nelec, self.nelec)) == 1] = 0

        if axis is None:
            return mat_cpy.sum()
        else:
            return mat_cpy.sum(axis)
