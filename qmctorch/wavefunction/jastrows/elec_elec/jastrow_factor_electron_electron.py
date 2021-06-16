import torch
from torch import nn
from ..distance.electron_electron_distance import ElectronElectronDistance
from .orbital_dependent_jastrow_kernel import OrbitalDependentJastrowKernel


class JastrowFactorElectronElectron(nn.Module):

    def __init__(self, nup, ndown,
                 jastrow_kernel,
                 kernel_kwargs={},
                 orbital_dependent_kernel=False,
                 number_of_orbitals=None,
                 scale=False, scale_factor=0.6,
                 cuda=False):
        """Electron-Electron Jastrow factor.

        .. math::
            J = \\prod_{i<j} \\exp(\\text{Kernel}(r_{ij}))

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            jastrow_kernel (kernel): class of a electron-electron Jastrow kernel
            kernel_kwargs (dict, optional): keyword argument of the kernel. Defaults to {}.
            orbital_dependent_kernel (bool, optional): Make the kernel orbital dependent. Defaults to False.
            number_of_orbitals (int, optional): number of orbitals for orbital dependent kernels. Defaults to None.
            scale (bool, optional): use scaled electron-electron distance. Defaults to False.
            scale_factor (float, optional): scaling factor. Defaults to 0.6.
            cuda (bool, optional): use cuda. Defaults to False.
        """

        super().__init__()

        self.nup = nup
        self.ndown = ndown
        self.nelec = nup + ndown
        self.ndim = 3

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

        self.requires_autograd = True

        # kernel function
        if orbital_dependent_kernel:
            self.jastrow_kernel = OrbitalDependentJastrowKernel(
                nup, ndown, number_of_orbitals, cuda, jastrow_kernel, kernel_kwargs)
        else:
            self.jastrow_kernel = jastrow_kernel(
                nup, ndown, cuda, **kernel_kwargs)
            self.requires_autograd = self.jastrow_kernel.requires_autograd

        # mask to extract the upper diag of the matrices
        self.mask_tri_up, self.index_col, self.index_row = self.get_mask_tri_up()

        # elec-elec distances
        self.edist = ElectronElectronDistance(self.nelec, self.ndim,
                                              scale=scale,
                                              scale_factor=scale_factor)

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
            inp (torch.tensor): input matrices (nbatch, n, n)

        Returns:
            torch.tensor: triangular up element (nbatch, -1)
        """
        nbatch = inp.shape[0]
        return inp.masked_select(self.mask_tri_up).view(nbatch, -1)

    def get_edist_unique(self, pos, derivative=0):
        """Get the unique elements of the electron-electron distance matrix.

        Args:
            pos (torch.tensor): positions of the electrons (Nbatch, Nelec*3)
            derivative(int, optional): order of the derivative

        Returns:
            torch.tensor: unique values of the electron-electron distance matrix
        """

        if derivative == 0:
            return self.extract_tri_up(self.edist(pos))

        elif derivative == 1:
            nbatch = pos.shape[0]
            return self.extract_tri_up(self.edist(
                pos, derivative=1)).view(nbatch, 3, -1)

        elif derivative == 2:
            nbatch = pos.shape[0]
            return self.extract_tri_up(self.edist(
                pos, derivative=2)).view(nbatch, 3, -1)

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

        r = self.get_edist_unique(pos)
        kern_vals = self.jastrow_kernel(r)
        jast = torch.exp(kern_vals.sum(-1)).unsqueeze(-1)

        if derivative == 0:
            return jast

        elif derivative == 1:
            dr = self.get_edist_unique(pos, derivative=1)
            return self.jastrow_factor_derivative(r, dr, jast, sum_grad)

        elif derivative == 2:

            dr = self.get_edist_unique(pos, derivative=1)
            d2r = self.get_edist_unique(pos, derivative=2)

            return self.jastrow_factor_second_derivative(r, dr, d2r, jast)

        elif derivative == [0, 1, 2]:

            dr = self.get_edist_unique(pos, derivative=1)
            d2r = self.get_edist_unique(pos, derivative=2)

            return(jast,
                   self.jastrow_factor_derivative(
                       r, dr, jast, sum_grad),
                   self.jastrow_factor_second_derivative(r, dr, d2r, jast))

    def jastrow_factor_derivative(self, r, dr, jast, sum_grad):
        """Compute the value of the derivative of the Jastrow factor

        Args:
            r (torch.tensor): distance matrix Nbatch x Nelec x Nelec
            dr (torch.tensor): derivative of the distance matrix Nbatch x Nelec x Nelec x 3
            jast (torch.tensor): values of the jastrow elements
                                 Nbatch x Nelec x Nelec
            sum_grad (bool): return the sum of the gradient or the individual components

        Returns:
            torch.tensor: gradient of the jastrow factors
                          Nbatch x Nelec x Ndim
        """

        if sum_grad:

            djast = self.jastrow_kernel.compute_derivative(
                r, dr).sum(-2)
            djast = djast * jast

            # might cause problems with backward cause in place operation
            out_shape = list(djast.shape[:-1]) + [self.nelec]
            out = torch.zeros(out_shape).to(self.device)
            out.index_add_(-1, self.index_row, djast)
            out.index_add_(-1, self.index_col, -djast)

        else:

            djast = self.jastrow_kernel.compute_derivative(
                r, dr)
            djast = djast * jast.unsqueeze(-1)

            # might cause problems with backward cause in place operation
            out_shape = list(djast.shape[:-1]) + [self.nelec]
            out = torch.zeros(out_shape).to(self.device)
            out.index_add_(-1, self.index_row, djast)
            out.index_add_(-1, self.index_col, -djast)

        return out

    def jastrow_factor_second_derivative(self, r, dr, d2r, jast):
        """Compute the value of the pure 2nd derivative of the Jastrow factor

        Args:
            r (torch.tensor): distance matrix Nbatch x Nelec x Nelec
            dr (torch.tensor): derivative of the distance matrix Nbatch x Nelec x Nelec x 3
            d2r (torch.tensor): 2nd derivative of the distance matrix Nbatch x Nelec x Nelec x 3
            jast (torch.tensor): values of the ajstrow elements
                                 Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: diagonal hessian of the jastrow factors
                          Nbatch x Nelec x Ndim
        """

        d2jast = self.jastrow_kernel.compute_second_derivative(
            r, dr, d2r).sum(-2)

        # might cause problems with backward cause in place operation
        hess_shape = list(d2jast.shape[:-1]) + [self.nelec]
        hess_jast = torch.zeros(hess_shape).to(self.device)
        hess_jast.index_add_(-1, self.index_row, d2jast)
        hess_jast.index_add_(-1, self.index_col, d2jast)

        # mixed terms
        djast = self.jastrow_kernel.compute_derivative(r, dr)

        # add partial derivative
        hess_jast = hess_jast + self.partial_derivative(djast)

        return hess_jast * jast

    def partial_derivative(self, djast):
        """Computes the partial derivative

        Args:
            djast (torch.tensor): values of the derivative of the jastrow kernels
        """

        out_shape = list(djast.shape[:-1]) + [self.nelec]
        out = torch.zeros(out_shape).to(self.device)
        out.index_add_(-1, self.index_row, djast)
        out.index_add_(-1, self.index_col, -djast)
        return (out**2).sum(-2)
