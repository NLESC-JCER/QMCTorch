import torch
from torch import nn
from .electron_distance import ElectronDistance
from .electron_nuclei_distance import ElectronNucleiDistance
import itertools


class ThreeBodyJastrowFactorBase(nn.Module):

    def __init__(self, nup, ndown, atomic_pos, cuda=False):
        r"""Base class for two body jastrow of the form:

        .. math::
            J = \prod_{i<j} \exp(B(r_{rij}))

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super(ThreeBodyJastrowFactorBase, self).__init__()

        self.nup = nup
        self.ndown = ndown
        self.nelec = nup + ndown
        self.atoms = atomic_pos
        self.natoms = atomic_pos.shape[0]
        self.ndim = 3

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

        self.mask_tri_up, self.index_col, self.index_row = self.get_mask_tri_up()
        self.index_elec = [
            self.index_col.tolist(), self.index_row.tolist()]

        self.elel_dist = ElectronDistance(self.nelec, self.ndim)
        self.elnu_dist = ElectronNucleiDistance(
            self.nelec, self.atoms, self.ndim)

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

    def extract_tri_up(self, inp):
        r"""extract the upper triangular elements

        Args:
            input (torch.tensor): input matrices (..., nelec, nelec)

        Returns:
            torch.tensor: triangular up element (..., nelec_pair)
        """
        shape = list(inp.shape)
        out = inp.masked_select(self.mask_tri_up)
        return out.view(*(shape[:-2] + [-1]))

    def extract_elec_nuc_dist(self, en_dist):
        r"""Organize the elec nuc distances

        Args:
            en_dist (torch.tensor): electron-nuclei distances 
                                    nbatch x nelec x natom or
                                    nbatch x 3 x nelec x natom (dr)

        Returns:
            torch.tensor: nbatch x natom x nelec_pair x 2 or
            torch.tensor: nbatch x 3 x natom x nelec_pair x 2 (dr)
        """
        out = en_dist[..., self.index_elec, :]
        if en_dist.ndim == 3:
            return out.permute(0, 3, 2, 1)
        elif en_dist.ndim == 4:
            return out.permute(0, 1, 4, 3, 2)
        else:
            raise ValueError(
                'elec-nuc distance matrix should have 3 or 4 dim')

    def assemble_dist(self, pos):
        """Assemle the different distances for easy calculations

        Args:
            pos (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim

        Returns:
            torch.tensor : nbatch, natom, nelec_pair, 3

        """

        # get the elec-elec distance matrix
        ree = self.extract_tri_up(self.elel_dist(pos))
        ree = ree.unsqueeze(1).unsqueeze(-1)
        ree = ree.repeat(1, self.natoms, 1, 1)

        # get the elec-nuc distance matrix
        ren = self.extract_elec_nuc_dist(self.elnu_dist(pos))

        # cat both
        return torch.cat((ren, ree), -1)

    def assemble_dist_deriv(self, pos, derivative=1):
        """Assemle the different distances for easy calculations

        Args:
            pos (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim

        Returns:
            torch.tensor : nbatch, 3 x natom, nelec_pair, 3

        """

        # get the elec-elec distance derivative
        dree = self.elel_dist(pos, derivative)
        dree = self.extract_tri_up(dree)
        dree = dree.unsqueeze(2).unsqueeze(-1)
        dree = dree.repeat(1, 1, self.natoms, 1, 1)

        # get the elec-nuc distance derivative
        dren = self.elnu_dist(pos, derivative)
        dren = self.extract_elec_nuc_dist(dren)

        # assemble
        return torch.cat((dren, dree), -1)

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

        r = self.assemble_dist(pos)
        jast = self._get_jastrow_elements(r)

        if derivative == 0:
            return jast.view(nbatch, -1).prod(-1).unsqueeze(-1)

        elif derivative == 1:
            dr = self.assemble_dist_deriv(pos, 1)
            return self._jastrow_derivative(r, dr, jast, jacobian)

        elif derivative == 2:

            dr = self.assemble_dist_deriv(pos, 1)
            d2r = self.assemble_dist_deriv(pos, 2)

            return self._jastrow_second_derivative(r, dr, d2r, jast)

        elif derivative == [0, 1, 2]:

            dr = self.assemble_dist_deriv(pos, 1)
            d2r = self.assemble_dist_deriv(pos, 2)

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

            prod_val = jast.view(nbatch, -1).prod(-1).unsqueeze(-1)
            djast = self._get_der_jastrow_elements(r, dr).sum(-2)
            print(djast.shape)
            print(prod_val.shape)
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
        hess_jast = hess_jast + 2 * \
            self.partial_derivative(djast)

        return hess_jast * prod_val

    def partial_derivative(self, djast):
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
            return tmp[..., self.idx_col_perm].prod(-1).sum(-3).sum(-1)
        else:
            out_shape = list(
                djast.shape[:-2]) + [self.nelec]
            return torch.zeros(out_shape).to(self.device)
