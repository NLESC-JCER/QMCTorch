import torch
from torch import nn

from .norm_orbital import atomic_orbital_norm
from .radial_functions import (radial_gaussian, radial_gaussian_pure,
                               radial_slater, radial_slater_pure)
from .spherical_harmonics import Harmonics
from .atomic_orbitals import AtomicOrbitals
from ..jastrows.distance.electron_electron_distance import ElectronElectronDistance


class AtomicOrbitalsBackFlow(AtomicOrbitals):

    def __init__(self, mol, cuda=False):
        """Computes the value of atomic orbitals

        Args:
            mol (Molecule): Molecule object
            cuda (bool, optional): Turn GPU ON/OFF Defaults to False.
        """

        super().__init__(mol, cuda)
        dtype = torch.get_default_dtype()
        self.backflow_weight = nn.Parameter(torch.as_tensor([0.1]))
        self.backflow_weight = nn.Parameter(
            torch.rand(mol.nelec, mol.nelec))
        self.edist = ElectronElectronDistance(mol.nelec)

    def _to_device(self):
        """Export the non parameter variable to the device."""

        self.device = torch.device('cuda')
        self.to(self.device)
        attrs = ['bas_n', 'bas_coeffs',
                 'nshells', 'norm_cst', 'index_ctr',
                 'backflow_weight']
        for at in attrs:
            self.__dict__[at] = self.__dict__[at].to(self.device)

    def forward(self, pos, derivative=[0], sum_grad=True, sum_hess=True, one_elec=False):
        r"""Computes the values of the atomic orbitals.

        .. math::
            \phi_i(r_j) = \sum_n c_n \\text{Rad}^{i}_n(r_j) \\text{Y}^{i}_n(r_j)

        where Rad is the radial part and Y the spherical harmonics part.
        It is also possible to compute the first and second derivatives

        .. math::
            \\nabla \phi_i(r_j) = \\frac{d}{dx_j} \phi_i(r_j) + \\frac{d}{dy_j} \phi_i(r_j) + \\frac{d}{dz_j} \phi_i(r_j) \n
            \\text{grad} \phi_i(r_j) = (\\frac{d}{dx_j} \phi_i(r_j), \\frac{d}{dy_j} \phi_i(r_j), \\frac{d}{dz_j} \phi_i(r_j)) \n
            \Delta \phi_i(r_j) = \\frac{d^2}{dx^2_j} \phi_i(r_j) + \\frac{d^2}{dy^2_j} \phi_i(r_j) + \\frac{d^2}{dz^2_j} \phi_i(r_j)

        Args:
            pos (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0,1,2,).
                                        Defaults to 0.
            sum_grad (bool, optional): Return the sum_grad (i.e. the sum of
                                       the derivatives) or the individual
                                       terms. Defaults to True.
                                       False only for derivative=1

            one_elec (bool, optional): if only one electron is in input

        Returns:
            torch.tensor: Value of the AO (or their derivatives) \n
                          size : Nbatch, Nelec, Norb (sum_grad = True) \n
                          size : Nbatch, Nelec, Norb, Ndim (sum_grad = False)

        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> ao = AtomicOrbitals(mol)
            >>> pos = torch.rand(100,6)
            >>> aovals = ao(pos)
            >>> daovals = ao(pos,derivative=1)
        """

        if not isinstance(derivative, list):
            derivative = [derivative]

        if not sum_grad:
            assert(1 in derivative)

        if not sum_hess:
            assert(2 in derivative)

        if one_elec:
            nelec_save = self.nelec
            self.nelec = 1

        if derivative == [0]:
            ao = self._compute_ao_values(pos)

        elif derivative == [1]:
            ao = self._compute_first_derivative_ao_values(
                pos, sum_grad)

        elif derivative == [2]:
            ao = self._compute_second_derivative_ao_values(
                pos, sum_hess)

        elif derivative == [0, 1, 2]:
            ao = self._compute_all_backflow_ao_values(pos)

        else:
            raise ValueError(
                'derivative must be 0, 1, 2 or [0, 1, 2], got ', derivative)

        if one_elec:
            self.nelec = nelec_save

        return ao

    def _compute_first_derivative_ao_values(self, pos, sum_grad):
        """Compute the value of the derivative of the ao from the xyx and r tensor

        Args:
            pos (torch.tensor): position of each elec size Nbatch, NelexNdim
            sum_grad (boolean): return the sum_grad (True) or gradient (False)

        Returns:
            torch.tensor: derivative of atomic orbital values
                          size (Nbatch, Nelec, Norb) if sum_grad
                          size (Nbatch, Nelec, Norb, Ndim) if sum_grad=False
        """
        grad = self._compute_gradient_backflow_ao_values(pos)

        if sum_grad:
            _, nbatch, nelec, norb = grad.shape
            grad = grad.reshape(nelec, 3, nbatch, nelec, norb).sum(1)

        return grad

    def _compute_gradient_backflow_ao_values(self, pos, grad_ao=None):
        """Compute the jacobian of the backflow ao fromn xyz tensor

        Args:
            pos ([type]): [description]

        Returns:
            torch.tensor derivative of atomic orbital values
                          size ([Nelec*Ndim], Nbatch, Nelec, Norb)
        """

        # get the grad vals : size Nbatch x Nelec x Norb x Ndim
        # the derivative are : d \phi_i(q_j) / d \tilde{x}_j
        # i.e. wrt to the back flow coordinates
        if grad_ao is None:
            grad_ao = self._compute_gradient_ao_values(pos)

        # permute the grad to Nbatch x Ndim x 1 x Nelec x 1 x Norb
        grad_ao = grad_ao.permute(0, 3, 1, 2)
        grad_ao = grad_ao.unsqueeze(2).unsqueeze(4)

        # compute the derivative of the bf positions wrt to the original pos
        # Nbatch x Ndim x Ndim x Nelec x Nelec x 1
        dbf = self._backflow_derivative(pos).unsqueeze(-1)

        # compute backflow : Nbatch x Ndim x Nelec x Nelec x Norb
        grad_ao = (grad_ao * dbf).sum(1)

        # permute to have Nelec x Ndim x Nbatch x Nelec x Norb
        grad_ao = grad_ao.permute(3, 1, 0, 2, 4)

        # collapse the first two dim [Nelec*Ndim] x Nbatch x Nelec x Norb
        grad_ao = grad_ao.reshape(-1, *(grad_ao.shape[2:]))

        return grad_ao

    def _compute_second_derivative_ao_values(self, pos, sum_hess):
        """Compute the value of the 2nd derivative of the ao from the xyx and r tensor

        Args:
            pos (torch.tensor): position of each elec size Nbatch, NelexNdim
            sum_hess (boolean): return the sum_hess (True) or diag hess (False)

        Returns:
            torch.tensor: derivative of atomic orbital values
                          size (Nbatch, Nelec, Norb) if sum_grad
                          size (Nbatch, Nelec, Norb, Ndim) if sum_grad=False
        """
        hess = self._compute_diag_hessian_backflow_ao_values(pos)

        if sum_hess:
            _, nbatch, nelec, norb = hess.shape
            hess = hess.reshape(nelec, 3, nbatch, nelec, norb).sum(1)

        return hess

    def _compute_diag_hessian_backflow_ao_values(self, pos, hess_ao=None, grad_ao=None):
        """Compute the laplacian of the backflow ao fromn xyz tensor

        Args:
            pos ([type]): [description]

        Returns:
            torch.tensor derivative of atomic orbital values
                          size (Nbatch, Nelec, Nelec, Norb)
        """
        # compute the lap size Nbatch x Nelec x Norb x 3
        if hess_ao is None:
            hess_ao = self._compute_diag_hessian_ao_values(pos)

        if grad_ao is None:
            grad_ao = self._compute_gradient_ao_values(pos)

        # permute the grad to Ndim x Nbatch x Nelec x Norb
        grad_ao = grad_ao.permute(3, 0, 1, 2)

        # permute the grad to Ndim x Nbatch x Nelec x Norb
        hess_ao = hess_ao.permute(3, 0, 1, 2)

        # compute the derivative of the bf positions wrt to the original pos
        # Ndim x Nbatch x Nelec x Nelec
        dbf = self._backflow_derivative(pos)
        dbf = dbf.permute(1, 0, 2, 3)

        # compute the 2nd derivative of the bf positions wrt to the original pos
        # Ndim x Nbatch x Nelec x Nelec
        d2bf = self._backflow_second_derivative(pos)
        d2bf = d2bf.permute(1, 0, 2, 3)

        # compute the back flow second der
        # I don't get it that should be dbf**2 !!!
        # WEIRD !
        hess_ao = hess_ao[..., None] @ (dbf)[..., None, :]

        # compute the backflow grad
        hess_ao += grad_ao[..., None] @ d2bf[..., None, :]

        # permute to have Nelec x Ndim x Nbatch x Nelec x Norb
        hess_ao = hess_ao.permute(2, 0, 1, 4, 3)
        # hess_ao = hess_ao.permute(4, 0, 1, 2, 3)

        # collapse the first two dim [Nelec*Ndim] x Nbatch x Nelec x Norb
        hess_ao = hess_ao.reshape(-1, *(hess_ao.shape[2:]))

        return hess_ao

    def _compute_all_backflow_ao_values(self, pos):
        """Compute the ao, gradient, laplacian of the ao from the xyx and r tensor

        Args:
            pos (torch.tensor): position of each elec size Nbatch, Nelec x Ndim

        Returns:
            tuple(): (ao, grad and lapalcian) of atomic orbital values
                     ao size (Nbatch, Nelec, Norb)
                     dao size (Nbatch, Nelec, Norb, Ndim)
                     d2ao size (Nbatch, Nelec, Norb)

        """

        xyz, r = self._process_position(pos)

        R, dR, d2R = self.radial(r, self.bas_n, self.bas_exp,
                                 xyz=xyz, derivative=[0, 1, 2],
                                 sum_grad=False, sum_hess=False)

        Y, dY, d2Y = self.harmonics(xyz,
                                    derivative=[0, 1, 2],
                                    sum_grad=False, sum_hess=False)

        # vals of the bf ao
        ao = self._ao_kernel(R, Y)

        # grad kernel of the bf ao
        grad_ao = self._gradient_kernel(R, dR, Y, dY)

        # diag hess kernel of the bf ao
        hess_ao = self._diag_hessian_kernel(
            R, dR, d2R, Y, dY, d2Y)

        # compute the bf ao
        hess_ao = self._compute_diag_hessian_backflow_ao_values(
            pos, hess_ao=hess_ao, grad_ao=grad_ao)

        # compute the bf grad
        grad_ao = self._compute_gradient_backflow_ao_values(
            pos, grad_ao=grad_ao)

        return (ao, grad_ao, hess_ao)

    def _process_position(self, pos):
        """Computes the positions/distance bewteen elec/orb

        Args:
            pos (torch.tensor): positions of the walkers Nbat, NelecxNdim

        Returns:
            torch.tensor, torch.tensor: positions of the elec wrt the bas
                                        (Nbatch, Nelec, Norb, Ndim)
                                        distance between elec and bas
                                        (Nbatch, Nelec, Norb)
        """
        # get the elec-atom vectrors/distances
        xyz, r = self._elec_atom_dist(pos)

        # repeat/interleave to get vector and distance between
        # electrons and orbitals
        return (xyz.repeat_interleave(self.nshells, dim=2),
                r.repeat_interleave(self.nshells, dim=2))

    def _elec_atom_dist(self, pos):
        """Computes the positions/distance bewteen elec/atoms

        Args:
            pos (torch.tensor): positions of the walkers Nbat, NelecxNdim

        Returns:
            torch.tensor, torch.tensor: positions of the elec wrt the bas
                                        (Nbatch, Nelec, Natom, Ndim)
                                        distance between elec and bas
                                        (Nbatch, Nelec, Natom)
        """

        # compute the back flow positions
        bf_pos = self._backflow(pos)

        # compute the vectors between electrons and atoms
        xyz = (bf_pos.view(-1, self.nelec, 1, self.ndim) -
               self.atom_coords[None, ...])

        # distance between electrons and atoms
        r = torch.sqrt((xyz*xyz).sum(3))

        return xyz, r

    def _backflow(self, pos):
        """transform the position of the electrons

        .. math:
            \\bold{q}_i = \\bold{r}_i + \\sum_{j\neq i} \\eta(r_{ij}) (\\bold{r}_i - \\bold{r}_j)

        Args:
            pos (torch.tensor): original positions Nbatch x [Nelec*Ndim]

        Returns:
            torch.tensor: transformed positions Nbatch x [Nelec*Ndim]
        """

        # compute the difference
        # Nbatch x Nelec x Nelec x 3
        delta_ee = self.edist.get_difference(
            pos.reshape(-1, self.nelec, self.ndim))

        # compute the backflow function
        # Nbatch x Nelec x Nelec
        bf_kernel = self._backflow_kernel(self.edist(pos))

        # update pos
        pos = pos.reshape(-1, self.nelec, self.ndim) + \
            (bf_kernel.unsqueeze(-1) * delta_ee).sum(2)

        return pos.reshape(-1, self.nelec*self.ndim)

    def _backflow_derivative(self, pos):
        r"""Computes the derivative of the back flow elec positions
           wrt the initial positions of the electrons

        .. math::
            \\bold{q}_i = \\bold{r}_i + \\sum_{j\\neq i} \\eta(r_{ij}) (\\bold{r}_i - \\bold{r}_j)

        .. math::
            \\frac{d q_i}{d x_k} = \\delta_{ik} (1 + \\sum_{j\\neq i} \\frac{d \\eta(r_ij)}{d x_i} (x_i-x_j) + \\eta(r_ij))  +
                                   \\delta_{i\\neq k} (-\\frac{d \\eta(r_ik)}{d x_k} (x_i-x_k) - \\eta(r_ik))


        Args:
            pos (torch.tensor): orginal positions of the electrons Nbatch x [Nelec*Ndim]

        Returns:
            torch.tensor: d q_{i}/d x_k  with :
                          q_{i} bf position of elec i
                          x_k original coordinate of the kth elec
                          Nelec x  Nbatch x Nelec x Norb x Ndim
        """

        # ee dist matrix : Nbatch x  Nelec x Nelec
        ree = self.edist(pos)
        nbatch, nelec, _ = ree.shape

        # derivative ee dist matrix : Nbatch x 3 x Nelec x Nelec
        # dr_ij / dx_i = - dr_ij / dx_j
        dree = self.edist(pos, derivative=1)

        # difference between elec pos
        # Nbatch, 3, Nelec, Nelec
        delta_ee = self.edist.get_difference(
            pos.reshape(nbatch, nelec, 3)).permute(0, 3, 1, 2)

        # backflow kernel : Nbatch x 1 x Nelec x Nelec
        bf = self._backflow_kernel(ree)

        # (d eta(r_ij) / d r_ij) (d r_ij/d beta_i)
        # derivative of the back flow kernel : Nbatch x 3 x Nelec x Nelec
        dbf = self._backflow_kernel_derivative(ree).unsqueeze(1)
        dbf = dbf * dree

        # (d eta(r_ij) / d beta_i) (alpha_i - alpha_j)
        # Nbatch x 3 x 3 x Nelec x Nelec
        dbf_delta_ee = dbf.unsqueeze(1) * delta_ee.unsqueeze(2)

        # compute the delta_ij * (1 + sum k \neq i eta(rik))
        # Nbatch x Nelec x Nelec (diagonal matrix)
        delta_ij_bf = torch.diag_embed(
            1 + bf.sum(-1), dim1=-1, dim2=-2)

        # eye 3x3 in 1x3x3x1x1
        I33 = torch.eye(3, 3).view(1, 3, 3, 1, 1)

        # compute the delta_ab * delta_ij * (1 + sum k \neq i eta(rik))
        # Nbatch x Ndim x Ndim x Nelec x Nelec (diagonal matrix)
        delta_ab_delta_ij_bf = I33 * \
            delta_ij_bf.view(nbatch, 1, 1, nelec, nelec)

        # compute sum_k df(r_ik)/dbeta_i (alpha_i - alpha_k)
        # Nbatch x Ndim x Ndim x Nelec x Nelec
        delta_ij_sum = torch.diag_embed(
            dbf_delta_ee.sum(-1), dim1=-1, dim2=-2)

        # compute delta_ab * f(rij)
        delta_ab_bf = I33 * bf.view(nbatch, 1, 1, nelec, nelec)

        # return Nbatch x Ndim(alpha) x Ndim(beta) x Nelec(i) x Nelec(j)
        # nbatch d alpha_i / d beta_j
        return delta_ab_delta_ij_bf + delta_ij_sum - dbf_delta_ee - delta_ab_bf

    def _backflow_second_derivative(self, pos):
        r"""Computes the seocnd derivative of the back flow elec positions
           wrt the initial positions of the electrons

        .. math::
            \\bold{q}_i = \\bold{r}_i + \\sum_{j\\neq i} \\eta(r_{ij}) (\\bold{r}_i - \\bold{r}_j)

        .. math::
            \\frac{d q_i}{d x_k} = \\delta_{ik} (1 + \\sum_{j\\neqi} \\frac{d \\eta(r_ij)}{d x_i} + \\eta(r_ij))  +
                                   \\delta_{i\\neq k} (-\\frac{d \\eta(r_ik)}{d x_k} - \\eta(r_ik))

        .. math::
            \\frac{d^2 q_i}{d x_k^2} = \\delta_{ik} (\\sum_{j\\neqi} \\frac{d^2 \\eta(r_ij)}{d x_i^2} + 2 \\frac{d \\eta(r_ij)}{d x_i})  +
                                       - \\delta_{i\\neq k} (\\frac{d^2 \\eta(r_ik)}{d x_k^2} + \\frac{d \\eta(r_ik)}{d x_k})


        Args:
            pos (torch.tensor): orginal positions of the electrons Nbatch x [Nelec*Ndim]

        Returns:
            torch.tensor: d q_{i}/d x_k  with :
                          q_{i} bf position of elec i
                          x_k original coordinate of the kth elec
                          Nelec x  Nbatch x Nelec x Norb x Ndim
        """

        # ee dist matrix :
        # Nbatch x  Nelec x Nelec
        ree = self.edist(pos)
        nbatch, nelec, _ = ree.shape

        # difference between elec pos
        # Nbatch, 3, Nelec, Nelec
        delta_ee = self.edist.get_difference(
            pos.reshape(nbatch, nelec, 3)).permute(0, 3, 1, 2)

        # derivative ee dist matrix  d r_{ij} / d x_i
        # Nbatch x 3 x Nelec x Nelec
        dree = self.edist(pos, derivative=1)

        # derivative ee dist matrix :  d2 r_{ij} / d2 x_i
        # Nbatch x 3 x Nelec x Nelec
        d2ree = self.edist(pos, derivative=2)

        # derivative of the back flow kernel : d eta(r_ij)/d r_ij
        # Nbatch x 1 x Nelec x Nelec
        dbf = self._backflow_kernel_derivative(ree).unsqueeze(1)

        # second derivative of the back flow kernel : d2 eta(r_ij)/d2 r_ij
        # Nbatch x 1 x Nelec x Nelec
        d2bf = self._backflow_kernel_second_derivative(
            ree).unsqueeze(1)

        # (d^2 eta(r_ij) / d r_ij^2) (d r_ij/d x_i)^2
        # + (d eta(r_ij) / d r_ij) (d^2 r_ij/d x_i^2)
        # Nbatch x 3 x Nelec x Nelec
        d2bf = (d2bf * dree * dree) + (dbf * d2ree)

        # (d eta(r_ij) / d r_ij) (d r_ij/d x_i)
        # Nbatch x 3 x Nelec x Nelec
        dbf = dbf * dree

        # eye matrix in dim x dim
        i33 = torch.eye(3, 3).reshape(1, 3, 3, 1, 1)

        # compute delta_ij delta_ab 2 sum_k dbf(ik) / dbeta_i
        term1 = 2 * i33 * \
            torch.diag_embed(
                dbf.sum(-1), dim1=-1, dim2=-2).reshape(nbatch, 1, 3, nelec, nelec)

        # (d2 eta(r_ij) / d2 beta_i) (alpha_i - alpha_j)
        # Nbatch x 3 x 3 x Nelec x Nelec
        d2bf_delta_ee = d2bf.unsqueeze(1) * delta_ee.unsqueeze(2)

        # compute sum_k d2f(r_ik)/d2beta_i (alpha_i - alpha_k)
        # Nbatch x Ndim x Ndim x Nelec x Nelec
        term2 = torch.diag_embed(
            d2bf_delta_ee.sum(-1), dim1=-1, dim2=-2)

        # compute delta_ab * df(rij)/dbeta_j
        term3 = 2 * i33 * dbf.reshape(nbatch, 1, 3, nelec, nelec)

        # return Nbatch x Ndim(alpha) x Ndim(beta) x Nelec(i) x Nelec(j)
        # nbatch d2 alpha_i / d2 beta_j
        return term1 + term2 + d2bf_delta_ee + term3

    def _backflow_kernel(self, ree):
        """Computes the backflow function:

        .. math:
            \\eta(r_{ij}) = \\frac{u}{r_{ij}}

        Args:
            r (torch.tensor): e-e distance Nbatch x Nelec x Nelec

        Returns:
            torch.tensor : f(r) Nbatch x Nelec x Nelec
        """
        return self.backflow_weight * ree * ree

        eye = torch.eye(self.nelec, self.nelec).to(self.device)
        mask = torch.ones_like(ree) - eye
        return self.backflow_weight * mask * (1./(ree+eye) - eye)

    def _backflow_kernel_derivative(self, ree):
        """Computes the derivative of the kernel function
            w.r.t r_{ij}
        .. math::
            \\frac{d}{dr_{ij} \\eta(r_{ij}) = -u r_{ij}^{-2}

        Args:
            ree (torch.tensor): e-e distance Nbatch x Nelec x Nelec

        Returns:
            torch.tensor : f'(r) Nbatch x Nelec x Nelec
        """

        return 2 * self.backflow_weight * ree

        eye = torch.eye(self.nelec, self.nelec).to(self.device)
        invree = (1./(ree+eye) - eye)
        return - self.backflow_weight * invree * invree

    def _backflow_kernel_second_derivative(self, ree):
        """Computes the derivative of the kernel function
            w.r.t r_{ij}
        .. math::
            \\frac{d}{dr_{ij} \\eta(r_{ij}) = -u r_{ij}^{-2}

        Args:
            ree (torch.tensor): e-e distance Nbatch x Nelec x Nelec

        Returns:
            torch.tensor : f''(r) Nbatch x Nelec x Nelec
        """
        return 2 * self.backflow_weight * torch.ones_like(ree)

        eye = torch.eye(self.nelec, self.nelec).to(self.device)
        invree = (1./(ree+eye) - eye)
        return 2 * self.backflow_weight * invree * invree * invree
