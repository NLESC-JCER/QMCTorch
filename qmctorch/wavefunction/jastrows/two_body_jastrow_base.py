import torch
from torch import nn
from .electron_distance import ElectronDistance
import itertools


class TwoBodyJastrowFactorBase(nn.Module):

    def __init__(self, nup, ndown, cuda=False):
        r"""Base class for two body jastrow of the form:

        .. math::
            J = \prod_{i<j} \exp(B(r_{rij}))

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super(TwoBodyJastrowFactorBase, self).__init__()

        self.nup = nup
        self.ndown = ndown
        self.nelec = nup + ndown
        self.ndim = 3

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

        self.mask_tri_up, self.index_col, self.index_row = self.get_mask_tri_up()

        self.edist = ElectronDistance(self.nelec, self.ndim)

        # choose the partial derivative method
        method = 'square'
        dict_method = {'col_perm': self._partial_derivative_col_perm,
                       'square': self._partial_dervivative_square}
        self.partial_derivative_method = dict_method[method]

        if method == 'col_perm':
            raise ValueError('col_perm is deprecated')
            self.idx_col_perm = torch.LongTensor(list(itertools.combinations(
                range(self.nelec-1), 2))).to(self.device)

    def get_static_weight(self):
        """Get the matrix of static weights

        Returns:
            torch.tensor: static weight (0.5 (0.25)for parallel(anti) spins
        """

        bup = torch.cat((0.25 * torch.ones(self.nup, self.nup), 0.5 *
                         torch.ones(self.nup, self.ndown)), dim=1)

        bdown = torch.cat((0.5 * torch.ones(self.ndown, self.nup), 0.25 *
                           torch.ones(self.ndown, self.ndown)), dim=1)

        static_weight = torch.cat((bup, bdown), dim=0).to(self.device)
        static_weight = static_weight.masked_select(self.mask_tri_up)

        return static_weight

    def _get_jastrow_elements(self, r):
        r"""Get the elements of the jastrow matrix :
        .. math::
            out_{i,j} = \exp{ U(r_{ij}) }

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the jastrow elements
                          Nbatch x Nelec x Nelec
        """
        raise NotImplementedError('Jastrow element not implemented')

    def _get_der_jastrow_elements(self, r, dr):
        """Get the elements of the derivative of the jastrow kernels
        wrt to the first electrons

        .. math::

            d B_{ij} / d k_i =  d B_{ij} / d k_j  = - d B_{ji} / d k_i

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the derivative of the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """
        raise NotImplementedError(
            'Jastrow derivative not implemented')

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
        raise NotImplementedError(
            'Jastrow second derivative not implemented')

    def get_mask_tri_up(self):
        r"""Get the mask to select the triangular up matrix

        Returns:
            torch.tensor: mask of the tri up matrix
        """
        mask = torch.zeros(self.nelec, self.nelec).type(
            torch.bool).to(self.device)
        index_col, index_row = [], []
        for i in range(self.nelec-1):
            for j in range(i+1, self.nelec):
                index_row.append(i)
                index_col.append(j)
                mask[i, j] = True

        index_col = torch.LongTensor(index_col).to(self.device)
        index_row = torch.LongTensor(index_row).to(self.device)
        return mask, index_col, index_row

    def extract_tri_up(self, input):
        r"""extract the upper triangular elements

        Args:
            input (torch.tensor): input matrices (nbatch, n, n)

        Returns:
            torch.tensor: triangular up element (nbatch, -1)
        """
        nbatch = input.shape[0]
        return input.masked_select(self.mask_tri_up).view(nbatch, -1)

    def _to_device(self):
        """Export the non parameter variable to the device."""

        self.device = torch.device('cuda')
        self.to(self.device)
        attrs = ['static_weight']
        for at in attrs:
            if at in self.__dict__:
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
                          derivative = 0  (Nmo) x Nbatch x 1
                          derivative = 1  (Nmo) x Nbatch x Nelec (for jacobian = True)
                          derivative = 1  (Nmo) x Nbatch x Ndim x Nelec (for jacobian = False)
                          derivative = 2  (Nmo) x Nbatch x Nelec
        """

        size = pos.shape
        assert size[1] == self.nelec * self.ndim
        nbatch = size[0]

        r = self.extract_tri_up(self.edist(pos))
        jast = self._get_jastrow_elements(r)

        if derivative == 0:
            return jast.prod(-1).unsqueeze(-1)

        elif derivative == 1:
            dr = self.extract_tri_up(self.edist(
                pos, derivative=1)).view(nbatch, 3, -1)
            return self._jastrow_derivative(r, dr, jast, jacobian)

        elif derivative == 2:

            dr = self.extract_tri_up(self.edist(
                pos, derivative=1)).view(nbatch, 3, -1)
            d2r = self.extract_tri_up(self.edist(
                pos, derivative=2)).view(nbatch, 3, -1)

            return self._jastrow_second_derivative(r, dr, d2r, jast)

        elif derivative == [0, 1, 2]:

            dr = self.extract_tri_up(self.edist(
                pos, derivative=1)).view(nbatch, 3, -1)
            d2r = self.extract_tri_up(self.edist(
                pos, derivative=2)).view(nbatch, 3, -1)

            return(jast.prod(-1).unsqueeze(-1),
                   self._jastrow_derivative(r, dr, jast, jacobian),
                   self._jastrow_second_derivative(r, dr, d2r, jast))

    def _jastrow_derivative(self, r, dr, jast, jacobian):
        """Compute the value of the derivative of the Jastrow factor

        Args:
            r (torch.tensor): ee distance matrix Nbatch x Nelec x Nelec
            jast (torch.tensor): values of the jastrow elements
                                 Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: gradient of the jastrow factors
                          Nbatch x Nelec x Ndim
        """
        nbatch = r.shape[0]
        if jacobian:

            prod_val = jast.prod(-1).unsqueeze(-1)
            djast = self._get_der_jastrow_elements(r, dr).sum(-2)
            djast = djast * prod_val

            # might cause problems with backward cause in place operation
            out_shape = list(djast.shape[:-1]) + [self.nelec]
            out = torch.zeros(out_shape).to(self.device)
            out.index_add_(-1, self.index_row, djast)
            out.index_add_(-1, self.index_col, -djast)

        else:

            prod_val = jast.prod(-1).unsqueeze(-1).unsqueeze(-1)
            djast = self._get_der_jastrow_elements(r, dr)
            djast = djast * prod_val

            # might cause problems with backward cause in place operation
            out_shape = list(djast.shape[:-1]) + [self.nelec]
            out = torch.zeros(out_shape).to(self.device)
            out.index_add_(-1, self.index_row, djast)
            out.index_add_(-1, self.index_col, -djast)

        return out

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
        nbatch = r.shape[0]

        # pure second derivative terms
        prod_val = jast.prod(-1).unsqueeze(-1)

        d2jast = self._get_second_der_jastrow_elements(
            r, dr, d2r).sum(-2)

        # might cause problems with backward cause in place operation
        hess_shape = list(d2jast.shape[:-1]) + [self.nelec]
        hess_jast = torch.zeros(hess_shape).to(self.device)
        hess_jast.index_add_(-1, self.index_row, d2jast)
        hess_jast.index_add_(-1, self.index_col, d2jast)

        # mixed terms
        djast = self._get_der_jastrow_elements(r, dr)

        # add partial derivative
        hess_jast = hess_jast + self.partial_derivative_method(djast)

        return hess_jast * prod_val

    def _single_index(self, i, j):
        """Compute the from the i,j index of a [nelec, nelec] matrix
        the index of a 1D array spanning the upper diagonal of the matrix.

            ij                  k

        00 01 02 03         . 0 1 2
        10 11 12 13         . . 3 4
        20 21 22 23         . . . 5
        31 31 32 33         . . . .

        """
        n = self.nelec
        return int((n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1)

    def _partial_derivative_col_perm(self, djast):
        """Get the product of the mixed second deriative terms using column permuatation.

        .. math ::

            d B_{ij} / d x_i * d B_{kl} / d x_k

        Args:
            djast (torch.tensor): first derivative of the jastrow kernels

        Returns:
            torch.tensor:
        """

        if len(self.idx_col_perm) > 0:
            tmp_shape = list(
                djast.shape[:-1]) + [self.nelec, self.nelec-1]
            tmp = torch.zeros(tmp_shape).to(self.device)
            tmp[..., self.index_row, self.index_col-1] = djast
            tmp[..., self.index_col, self.index_row] = -djast
            return 2*tmp[..., self.idx_col_perm].prod(-1).sum(-3).sum(-1)
        else:
            out_shape = list(
                djast.shape[:-2]) + [self.nelec]
            return torch.zeros(out_shape).to(self.device)

    def _partial_dervivative_square(self, djast):
        """[summary]

        Args:
            djast ([type]): [description]
        """

        out_shape = list(djast.shape[:-1]) + [self.nelec]
        out = torch.zeros(out_shape).to(self.device)
        out.index_add_(-1, self.index_row, djast)
        out.index_add_(-1, self.index_col, -djast)
        return (out**2).sum(-2)
