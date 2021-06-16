import torch
from torch import nn
import torch
from torch.autograd import Variable, grad

from ..distance.electron_electron_distance import ElectronElectronDistance
from ..distance.electron_nuclei_distance import ElectronNucleiDistance


class JastrowFactorElectronElectronNuclei(nn.Module):

    def __init__(self, nup, ndown, atomic_pos,
                 jastrow_kernel,
                 kernel_kwargs={},
                 cuda=False):
        """Jastrow Factor of the elec-elec-nuc term:

        .. math::
            J =  \\exp\\left(  \\sum_A \\sum_{i<j} K(R_{iA}, r_{jA}, r_{rij}) \\right)

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super().__init__()

        self.nup = nup
        self.ndown = ndown
        self.nelec = nup + ndown

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

        self.atoms = atomic_pos.to(self.device)
        self.natoms = atomic_pos.shape[0]
        self.ndim = 3

        # kernel function
        self.jastrow_kernel = jastrow_kernel(nup, ndown,
                                             atomic_pos,
                                             cuda,
                                             **kernel_kwargs)

        # requires autograd to compute derivatives
        self.requires_autograd = self.jastrow_kernel.requires_autograd

        # index to extract tri up matrices
        self.mask_tri_up, self.index_col, self.index_row = self.get_mask_tri_up()
        self.index_elec = [
            self.index_row.tolist(), self.index_col.tolist()]

        # distance calculator
        self.elel_dist = ElectronElectronDistance(
            self.nelec, self.ndim)
        self.elnu_dist = ElectronNucleiDistance(
            self.nelec, self.atoms, self.ndim)

        # method to compute the second derivative
        # If False jastrow_factor_second_derivative will be used
        # this method  only works when the kernel does not multiply
        # the different terms e.g : k = f(r_ij) + g(R_iA) + g(R_jA)
        # For non-lienar kernels e.g. : k = f(r_ij) * g(R_iA) * g(R_jA)
        # auto_second_derivative must be set to True.
        self.auto_second_derivative = True

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
           the output has dimension  nbatch, 3 x natom, nelec_pair, 3
           the last dimension is composed of [r_{e_1n}, r_{e_2n}, r_{ee}]

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

    def forward(self, pos, derivative=0, sum_grad=True):
        """Compute the Jastrow factors.

        Args:
            pos (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0,1,2,).
                            Defaults to 0.
            sum_grad (bool, optional): Return the sum_grad (i.e. the sum of
                                       the derivatives) or the individual
                                       terms. Defaults to True.
                                       False only for derivative=1

        Returns:
            torch.tensor: value of the jastrow parameter for all confs
                          derivative = 0  (Nmo) x Nbatch x 1
                          derivative = 1  (Nmo) x Nbatch x Nelec (for sum_grad = True)
                          derivative = 1  (Nmo) x Nbatch x Ndim x Nelec (for sum_grad = False)
                          derivative = 2  (Nmo) x Nbatch x Nelec
        """

        size = pos.shape
        assert size[1] == self.nelec * self.ndim
        nbatch = size[0]

        r = self.assemble_dist(pos)
        kern_vals = self.jastrow_kernel(r)
        jast = torch.exp(kern_vals.view(nbatch, -1).sum(-1))

        if derivative == 0:
            return jast.unsqueeze(-1)

        elif derivative == 1:
            dr = self.assemble_dist_deriv(pos, 1)
            return self.jastrow_factor_derivative(r, dr, jast, sum_grad)

        elif derivative == 2:

            if self.auto_second_derivative:
                return self.jastrow_factor_second_derivative_auto(pos, jast=jast.unsqueeze(-1))

            else:
                dr = self.assemble_dist_deriv(pos, 1)
                d2r = self.assemble_dist_deriv(pos, 2)

                return self.jastrow_factor_second_derivative(r, dr, d2r, jast)

        elif derivative == [0, 1, 2]:

            dr = self.assemble_dist_deriv(pos, 1)
            djast = self.jastrow_factor_derivative(
                r, dr, jast, sum_grad)

            if self.auto_second_derivative:
                d2jast = self.jastrow_factor_second_derivative_auto(
                    pos, jast=jast.unsqueeze(-1))
            else:
                d2r = self.assemble_dist_deriv(pos, 2)
                d2jast = self.jastrow_factor_second_derivative(
                    r, dr, d2r, jast)

            return(jast.unsqueeze(-1), djast, d2jast)

        else:
            raise ValueError('Derivative value nor recognized')

    def jastrow_factor_derivative(self, r, dr, jast, sum_grad):
        """Compute the value of the derivative of the Jastrow factor

        Args:
            r (torch.tensor): ee distance matrix Nbatch x Nelec x Nelec
            jast (torch.tensor): values of the jastrow elements
                                 Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: gradient of the jastrow factors
                          Nbatch x Nelec x Ndim
        """

        if sum_grad:

            # derivative of the jastrow elements
            # nbatch x ndim x natom x nelec_pair x 3
            # last dim is (ria rja rij)
            djast = self.jastrow_kernel.compute_derivative(r, dr)

            # sum dim and atom
            djast = djast.sum([1, 2])

            # multiply with the product of jastrow el values
            djast = djast * jast.unsqueeze(-1).unsqueeze(-1)

            # create the output vector with size nbatch x nelec
            out_shape = list(djast.shape[:-2]) + [self.nelec]
            out = torch.zeros(out_shape).to(self.device)

            # add the elec-elec term
            out.index_add_(-1, self.index_row, djast[..., 2])
            out.index_add_(-1, self.index_col, -djast[..., 2])

            # add the elec-nuc terms
            out.index_add_(-1, self.index_row, djast[..., 0])
            out.index_add_(-1, self.index_col, djast[..., 1])

        else:

            # derivative of the jastrow elements
            # nbatch x ndim x natom x nelec_pair x 3
            # last dim is (ria rja rij)
            djast = self.jastrow_kernel.compute_derivative(r, dr)

            # sum atom
            djast = djast.sum(2)
            djast = djast * \
                jast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # might cause problems with backward cause in place operation
            out_shape = list(djast.shape[:-2]) + [self.nelec]
            out = torch.zeros(out_shape).to(self.device)

            # add electronic terms
            out.index_add_(-1, self.index_row, djast[..., 2])
            out.index_add_(-1, self.index_col, -djast[..., 2])

            # add elec-nuc terms
            out.index_add_(-1, self.index_row, djast[..., 0])
            out.index_add_(-1, self.index_col, djast[..., 1])

        return out

    def jastrow_factor_second_derivative(self, r, dr, d2r, jast):
        """Compute the value of the pure 2nd derivative of the Jastrow factor

        Args:
            r (torch.tensor): ee distance matrix Nbatch x Nelec x Nelec
            jast (torch.tensor): values of the ajstrow elements
                                 Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: diagonal hessian of the jastrow factors
                          Nbatch x Nelec x Ndim
        """

        # puresecond derivative of the jast el
        # nbatch x ndim x natom x nelec_pair x 3
        # last dim is (ria rja rij)
        d2jast = self.jastrow_kernel.compute_second_derivative(
            r, dr, d2r)

        # sum over the dim and the atom
        d2jast = d2jast.sum([1, 2])

        # might cause problems with backward cause in place operation
        hess_shape = list(d2jast.shape[:-2]) + [self.nelec]
        hess_jast = torch.zeros(hess_shape).to(self.device)

        # add elec-elec terms
        hess_jast.index_add_(-1, self.index_row, d2jast[..., 2])
        hess_jast.index_add_(-1, self.index_col, d2jast[..., 2])

        # add elec-nu terms
        hess_jast.index_add_(-1, self.index_row, d2jast[..., 0])
        hess_jast.index_add_(-1, self.index_col, d2jast[..., 1])

        # mixed terms
        djast = self.jastrow_kernel.compute_derivative(r, dr)

        # add partial derivative
        hess_jast = hess_jast + self.partial_derivative(djast)

        return hess_jast * jast.unsqueeze(-1)

    def partial_derivative(self, djast):
        """[summary]

        Args:
            djast ([type]): [description]
        """

        # create the output vector with size nbatch x nelec
        out_shape = list(djast.shape[:-2]) + [self.nelec]
        out = torch.zeros(out_shape).to(self.device)

        # add the elec-elec term
        out.index_add_(-1, self.index_row, djast[..., 2])
        out.index_add_(-1, self.index_col, -djast[..., 2])

        # add the elec-nuc terms
        out.index_add_(-1, self.index_row, djast[..., 0])
        out.index_add_(-1, self.index_col, djast[..., 1])

        return ((out.sum(2))**2).sum(1)

    def jastrow_factor_second_derivative_auto(self, pos, jast=None):
        """Compute the second derivative of the jastrow factor automatically.
        This is needed for complicate kernels where the partial derivatives of
        the kernels are difficult to organize in a total derivaitve e.e Boys-Handy

        Args:
            pos ([type]): [description]
        """

        def hess(out, pos):

            # compute the jacobian
            z = Variable(torch.ones_like(out))
            jacob = grad(out, pos,
                         grad_outputs=z,
                         only_inputs=True,
                         create_graph=True)[0]

            # compute the diagonal element of the Hessian
            z = Variable(torch.ones(jacob.shape[0])).to(self.device)
            hess = torch.zeros_like(jacob)

            for idim in range(jacob.shape[1]):

                tmp = grad(jacob[:, idim], pos,
                           grad_outputs=z,
                           only_inputs=True,
                           create_graph=True)[0]

                hess[:, idim] = tmp[:, idim]

            return hess

        nbatch = pos.shape[0]
        if jast is None:
            jast = self.forward(pos)
        return hess(jast, pos).view(nbatch, self.nelec, 3).sum(2)
