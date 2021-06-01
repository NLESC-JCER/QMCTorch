import torch

from .atomic_orbitals import AtomicOrbitals
from .backflow.orbital_dependent_backflow_transformation import OrbitalDependentBackFlowTransformation


class AtomicOrbitalsOrbitalDependentBackFlow(AtomicOrbitals):

    def __init__(self, mol, backflow_kernel, backflow_kernel_kwargs={}, cuda=False):
        """Computes the value of atomic orbitals

        Args:
            mol (Molecule): Molecule object
            cuda (bool, optional): Turn GPU ON/OFF Defaults to False.
        """

        super().__init__(mol, cuda)
        dtype = torch.get_default_dtype()
        self.backflow_trans = OrbitalDependentBackFlowTransformation(mol,
                                                                     backflow_kernel=backflow_kernel,
                                                                     backflow_kernel_kwargs=backflow_kernel_kwargs,
                                                                     cuda=cuda)

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

        elif derivative == [3]:
            ao = self._compute_mixed_second_derivative_ao_values(pos)

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
        # Nbatch x Ndim x Ndim x Nelec x Nelec x Norb
        dbf = self.backflow_trans(
            pos, derivative=1).permute(0, 2, 3, 4, 5, 1)

        # compute backflow : Nbatch x Ndim x Nelec x Nelec x Norb
        grad_ao = (grad_ao * dbf).sum(1)

        # permute to have Nelec x Ndim x Nbatch x Nelec x Norb
        # Note the idex of the electron (2,3) are inverted
        # compare to the single backflow trans case
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

    def _compute_diag_hessian_backflow_ao_values(self, pos, hess_ao=None, mixed_ao=None, grad_ao=None):
        """Compute the laplacian of the backflow ao fromn xyz tensor

        Args:
            pos ([type]): [description]

        Returns:
            torch.tensor derivative of atomic orbital values
                          size (Nbatch, Nelec, Nelec, Norb)
        """
        # compute the lap size Nbatch x Ndim x Ndim x Nelec x Norb
        if hess_ao is None:
            hess_ao = self._compute_diag_hessian_ao_values(pos)

        if mixed_ao is None:
            mixed_ao = self._compute_mixed_second_derivative_ao_values(
                pos)

        if grad_ao is None:
            grad_ao = self._compute_gradient_ao_values(pos)

        # permute the grad to Nbatch x Ndim x 1 x Nelec x 1 x Norb
        grad_ao = grad_ao.permute(0, 3, 1, 2)
        grad_ao = grad_ao.unsqueeze(2).unsqueeze(4)

        # permute the hess to Nbatch x Ndim x 1 x Nelec x 1 x Norb
        hess_ao = hess_ao.permute(0, 3, 1, 2)
        hess_ao = hess_ao.unsqueeze(2).unsqueeze(4)

        # permute the hess to Nbatch x Ndim x 1 x Nelec x 1 x Norb
        mixed_ao = mixed_ao.permute(0, 3, 1, 2)
        mixed_ao = mixed_ao.unsqueeze(2).unsqueeze(4)

        # compute the derivative of the bf positions wrt to the original pos
        # Nbatch x Ndim x Ndim x Nelec x Nelec x Norb
        dbf = self.backflow_trans(
            pos, derivative=1).permute(0, 2, 3, 4, 5, 1)

        # compute the derivative of the bf positions wrt to the original pos
        # Nbatch x Ndim x Ndim x Nelec x Nelec x Norb
        d2bf = self.backflow_trans(
            pos, derivative=2).permute(0, 2, 3, 4, 5, 1)

        # compute the back flow second der
        hess_ao = (hess_ao * (dbf*dbf)).sum(1)

        # compute the backflow grad
        hess_ao += (grad_ao * d2bf).sum(1)

        # compute the contribution of the mixed derivative
        hess_ao += 2*(mixed_ao *
                      dbf[:, [[0, 1], [0, 2], [1, 2]], ...].prod(2)).sum(1)

        # permute to have Nelec x Ndim x Nbatch x Nelec x Norb
        hess_ao = hess_ao.permute(3, 1, 0, 2, 4)

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
        xyz, r = self._elec_ao_dist(pos)

        if self.contract:
            # repeat/interleave to get vector and distance between
            # electrons and orbitals
            xyz = xyz.repeat_interleave(self.nctr_per_ao, dim=2)
            r = r.repeat_interleave(self.nctr_per_ao, dim=2)

        return xyz, r

    def _elec_ao_dist(self, pos):
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
        # Nbatch x Nao x Nelec*Ndim
        bf_pos = self.backflow_trans(pos)
        nbatch, nao, _ = bf_pos.shape

        # reshape
        bf_pos = bf_pos.view(nbatch, nao, self.nelec, self.ndim)

        # permute to nbatch x nelec x nao x ndim
        bf_pos = bf_pos.permute(0, 2, 1, 3)

        # interleave the atomic positions
        # nao x ndim
        atom_coords = self.atom_coords.repeat_interleave(
            self.nao_per_atom, dim=0)

        # compute the vectors between electrons and atoms
        # nbatch x nelec x nao x ndim
        xyz = (bf_pos-atom_coords)

        # distance between electrons and atoms
        # nbatch x nelec x nao
        r = torch.sqrt((xyz*xyz).sum(3))

        return xyz, r
