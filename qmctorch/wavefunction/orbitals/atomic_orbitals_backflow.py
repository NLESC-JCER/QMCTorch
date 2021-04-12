import torch
from torch import nn

from .norm_orbital import atomic_orbital_norm
from .radial_functions import (radial_gaussian, radial_gaussian_pure,
                               radial_slater, radial_slater_pure)
from .spherical_harmonics import Harmonics
from .atomic_orbitals import AtomicOrbitals


class AtomicOrbitalsBackFlow(AtomicOrbitals):

    def __init__(self, mol, cuda=False):
        """Computes the value of atomic orbitals

        Args:
            mol (Molecule): Molecule object
            cuda (bool, optional): Turn GPU ON/OFF Defaults to False.
        """

        super().__init__(mol, cuda)
        dtype = torch.get_default_dtype()
        self.backflow_weights = nn.Parameter(
            torch.eye(self.nelec, self.nelec))

    def _to_device(self):
        """Export the non parameter variable to the device."""

        self.device = torch.device('cuda')
        self.to(self.device)
        attrs = ['bas_n', 'bas_coeffs',
                 'nshells', 'norm_cst', 'index_ctr',
                 'backflow_weights']
        for at in attrs:
            self.__dict__[at] = self.__dict__[at].to(self.device)

    def forward(self, pos, derivative=[0], jacobian=True, one_elec=False):
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
            jacobian (bool, optional): Return the jacobian (i.e. the sum of
                                       the derivatives) or the individual
                                       terms. Defaults to True.
                                       False only for derivative=1

            one_elec (bool, optional): if only one electron is in input

        Returns:
            torch.tensor: Value of the AO (or their derivatives) \n
                          size : Nbatch, Nelec, Norb (jacobian = True) \n
                          size : Nbatch, Nelec, Norb, Ndim (jacobian = False)

        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> ao = AtomicOrbitals(mol)
            >>> pos = torch.rand(100,6)
            >>> aovals = ao(pos)
            >>> daovals = ao(pos,derivative=1)
        """

        if not isinstance(derivative, list):
            derivative = [derivative]

        if not jacobian:
            assert(1 in derivative)

        if one_elec:
            nelec_save = self.nelec
            self.nelec = 1

        if derivative == [0]:
            ao = self._compute_ao_values(pos)

        elif derivative == [1]:
            ao = self._compute_first_derivative_backflow_ao_values(
                pos, jacobian)

        elif derivative == [2]:
            ao = self._compute_laplacian_backflow_ao_values(pos)

        elif derivative == [0, 1, 2]:
            ao = self._compute_all_backflow_ao_values(pos)

        else:
            raise ValueError(
                'derivative must be 0, 1, 2 or [0, 1, 2], got ', derivative)

        if one_elec:
            self.nelec = nelec_save

        return ao

    def _compute_first_derivative_ao_values(self, pos, jacobian):
        """Compute the value of the derivative of the ao from the xyx and r tensor

        Args:
            pos (torch.tensor): position of each elec size Nbatch, NelexNdim
            jacobian (boolean): return the jacobian (True) or gradient (False)

        Returns:
            torch.tensor: derivative of atomic orbital values
                          size (Nbatch, Nelec, Norb) if jacobian
                          size (Nbatch, Nelec, Norb, Ndim) if jacobian=False
        """
        if jacobian:
            return self._compute_jacobian_backflow_ao_values(pos)
        else:
            return self._compute_gradient_backflow_ao_values(pos)

    def _compute_jacobian_backflow_ao_values(self, pos, jac_ao=None):
        """Compute the jacobian of the backflow ao fromn xyz tensor

        Args:
            pos ([type]): [description]

        Returns:
            torch.tensor derivative of atomic orbital values
                          size (Nelec, Nbatch, Nelec, Norb)
        """
        # get the grad vals : size Nbatch x Nelec x Norb
        if jac_ao is None:
            jac_ao = self._compute_jacobian_ao_values(pos)

        # compute backflow : Nbatch x Nelec X Norb x Nelec
        jac_ao = jac_ao[..., None] @ self.backflow_weights[:, None, :]

        # permute to Nelec, Nbatch, Nelec, Norb
        return jac_ao.permute(3, 0, 1, 2)

    def _compute_gradient_backflow_ao_values(self, pos, grad_ao=None):
        """Compute the jacobian of the backflow ao fromn xyz tensor

        Args:
            pos ([type]): [description]

        Returns:
            torch.tensor derivative of atomic orbital values
                          size ([Nelec*Ndim], Nbatch, Nelec, Norb)
        """

        # get the grad vals : size Nbatch x Nelec x Norb x Ndim
        if grad_ao is None:
            grad_ao = self._compute_gradient_ao_values(pos)

        # permute the grad to Ndim x Nbatch x Nelec x Norb
        grad_ao = grad_ao.permute(0, 3, 1, 2)

        # compute outer product : Ndim, Nbatch, Nelec, Norb, Nelec
        grad_ao = grad_ao[...,
                          None] @ self.backflow_weights[:, None, :]

        # permute to get Nelec Ndim, Nbatch, Nelec, Norb
        grad_ao = grad_ao.permute(0, 1, 2, 3, 4)

        # collapse the first two dim [Nelec*Ndim] x Nbatch x Nelec x Norb
        grad_ao = grad_ao.reshape(-1, *(grad_ao.shape[2:]))

        return grad_ao

    def _compute_laplacian_backflow_ao_values(self, pos, lap_ao=None):
        """Compute the jacobian of the backflow ao fromn xyz tensor

        Args:
            pos ([type]): [description]

        Returns:
            torch.tensor derivative of atomic orbital values
                          size (Nbatch, Nelec, Nelec, Norb)
        """
        # compute the lap size Nbatch x Nelec x Norb
        if lap_ao is None:
            lap_ao = self._compute_laplacian_ao_values(pos)

        # compute square of weights
        bfw_squared = self.backflow_weights*self.backflow_weights

        # compute backflow : Nbatch x Nelec X Norb x Nelec
        lap_ao = lap_ao[..., None] @ bfw_squared[:, None, :]

        # permute to Nelec, Nbatch, Nelec, Norb
        return lap_ao.permute(3, 0, 1, 2)

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
                                 jacobian=False)

        Y, dY, d2Y = self.harmonics(xyz,
                                    derivative=[0, 1, 2],
                                    jacobian=False)

        # vals of the bf_ao
        ao = self._ao_kernel(R, Y)

        # grad of the bf ao
        grad_ao = self._gradient_kernel(R, dR, Y, dY)
        grad_ao = self._compute_gradient_backflow_ao_values(
            xyz, grad_ao=grad_ao)

        # lap of the bf ao
        lap_ao = self._laplacian_kernel(R, dR, d2R, Y, dY, d2Y)
        lap_ao = self._compute_laplacian_backflow_ao_values(
            xyz, lap_ao=lap_ao)

        return (ao, grad_ao, lap_ao)

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
        # we should remove that line but that means
        # geometry optimization should update that
        self.bas_coords = self.atom_coords.repeat_interleave(
            self.nshells, dim=0)

        xyz = (pos.view(-1, self.nelec, 1, self.ndim) -
               self.bas_coords[None, ...])

        # backflow transformation
        xyz = (xyz.permute(0, 2, 3, 1) @
               self.backflow_weights).permute(0, 3, 1, 2)

        r = torch.sqrt((xyz*xyz).sum(3))

        return xyz, r
