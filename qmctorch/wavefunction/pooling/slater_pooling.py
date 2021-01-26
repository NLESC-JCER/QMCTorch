import torch
from torch import nn
import operator as op

from ...utils import bdet2, btrace
from .orbital_configurations import get_excitation, get_unique_excitation
from .orbital_projector import ExcitationMask, OrbitalProjector


class SlaterPooling(nn.Module):

    """Applies a slater determinant pooling in the active space."""

    def __init__(self, config_method, configs, mol, cuda=False):
        """Computes the Sater determinants

        Args:
            config_method (str): method used to define the config
            configs (tuple): configuratin of the electrons
            mol (Molecule): Molecule instance
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.

        """
        super(SlaterPooling, self).__init__()

        self.config_method = config_method

        self.configs = configs
        self.nconfs = len(configs[0])
        self.index_max_orb_up = self.configs[0].max().item() + 1
        self.index_max_orb_down = self.configs[1].max().item() + 1

        self.excitation_index = get_excitation(configs)
        self.unique_excitation, self.index_unique_excitation = get_unique_excitation(
            configs)

        self.nmo = mol.basis.nmo
        self.nup = mol.nup
        self.ndown = mol.ndown
        self.nelec = self.nup + self.ndown

        self.orb_proj = OrbitalProjector(configs, mol, cuda=cuda)
        self.exc_mask = ExcitationMask(self.unique_excitation, mol,
                                       (self.index_max_orb_up,
                                        self.index_max_orb_down),
                                       cuda=cuda)

        self.device = torch.device('cpu')
        if cuda:
            self.device = torch.device('cuda')

    def forward(self, input):
        """Computes the values of the determinats

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo

        Returns:
            torch.tensor: slater determinants
        """
        if self.config_method.startswith('cas('):
            return self.det_explicit(input)
        else:
            return self.det_single_double(input)

    def get_slater_matrices(self, input):
        """Computes the slater matrices

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo


        Returns:
            (torch.tensor, torch.tensor): slater matrices of spin up/down
        """
        return self.orb_proj.split_orbitals(input)

    def det_explicit(self, input):
        """Computes the values of the determinants from the slater matrices

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo

        Returns:
            torch.tensor: slater determinants
        """

        mo_up, mo_down = self.get_slater_matrices(input)
        return (torch.det(mo_up) * torch.det(mo_down)).transpose(0, 1)

    def det_single_double(self, input):
        """Computes the determinant of ground state + single + double

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo

        Returns:
            torch.tensor: slater determinants
        """

        # compute the determinant of the unique single excitation
        det_unique_up, det_unique_down = self.det_unique_single_double(
            input)

        # returns the product of spin up/down required by each excitation
        return (det_unique_up[:, self.index_unique_excitation[0]] *
                det_unique_down[:, self.index_unique_excitation[1]])

    def det_ground_state(self, input):
        """Computes the SD of the ground state

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo
        """

        return (torch.det(input[:, :self.nup, :self.nup]),
                torch.det(input[:, self.nup:, :self.ndown]))

    def det_unique_single_double(self, input):
        """Computes the SD of single/double excitations

        The determinants of the single excitations
        are calculated from the ground state determinant and
        the ground state Slater matrices whith one column modified.
        See : Monte Carlo Methods in ab initio quantum chemistry
        B.L. Hammond, appendix B1


        Note : if the state on coonfigs are specified in order
        we end up with excitations that comes from a deep orbital, the resulting
        slater matrix has one column changed (with the new orbital) and several
        permutation. We therefore need to multiply the slater determinant
        by (-1)^nperm.


        .. math::

            MO = [ A | B ]
            det(Exc_{ij}) = (det(A) * A^{-1} * B)_{i,j}

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo

        """

        nbatch = input.shape[0]

        if not hasattr(self.exc_mask, 'index_unique_single_up'):
            self.exc_mask.get_index_unique_single()

        if not hasattr(self.exc_mask, 'index_unique_double_up'):
            self.exc_mask.get_index_unique_double()

        do_single = len(self.exc_mask.index_unique_single_up) != 0
        do_double = len(self.exc_mask.index_unique_double_up) != 0

        # occupied orbital matrix + det and inv on spin up
        Aup = input[:, :self.nup, :self.nup]
        detAup = torch.det(Aup)

        # occupied orbital matrix + det and inv on spin down
        Adown = input[:, self.nup:, :self.ndown]
        detAdown = torch.det(Adown)

        # store all the dets we need
        det_out_up = detAup.unsqueeze(-1).clone()
        det_out_down = detAdown.unsqueeze(-1).clone()

        # return the ground state
        if self.config_method == 'ground_state':
            return det_out_up, det_out_down

        # inverse of the
        invAup = torch.inverse(Aup)
        invAdown = torch.inverse(Adown)

        # virtual orbital matrices spin up/down
        Bup = input[:, :self.nup, self.nup:self.index_max_orb_up]
        Bdown = input[:, self.nup:,
                      self.ndown: self.index_max_orb_down]

        # compute the products of Ain and B
        mat_exc_up = (invAup @ Bup)
        mat_exc_down = (invAdown @ Bdown)

        if do_single:

            # determinant of the unique excitation spin up
            det_single_up = mat_exc_up.view(
                nbatch, -1)[:, self.exc_mask.index_unique_single_up]

            # determinant of the unique excitation spin down
            det_single_down = mat_exc_down.view(
                nbatch, -1)[:, self.exc_mask.index_unique_single_down]

            # multiply with ground state determinant
            # and account for permutation for deep excitation
            det_single_up = detAup.unsqueeze(-1) * \
                det_single_up.view(nbatch, -1)

            # multiply with ground state determinant
            # and account for permutation for deep excitation
            det_single_down = detAdown.unsqueeze(-1) * \
                det_single_down.view(nbatch, -1)

            # accumulate the dets
            det_out_up = torch.cat((det_out_up, det_single_up), dim=1)
            det_out_down = torch.cat(
                (det_out_down, det_single_down), dim=1)

        if do_double:

            # det of unique spin up double exc
            det_double_up = mat_exc_up.view(
                nbatch, -1)[:, self.exc_mask.index_unique_double_up]

            det_double_up = bdet2(
                det_double_up.view(nbatch, -1, 2, 2))

            det_double_up = detAup.unsqueeze(-1) * det_double_up

            # det of unique spin down double exc
            det_double_down = mat_exc_down.view(
                nbatch, -1)[:, self.exc_mask.index_unique_double_down]

            det_double_down = bdet2(
                det_double_down.view(nbatch, -1, 2, 2))

            det_double_down = detAdown.unsqueeze(-1) * det_double_down

            det_out_up = torch.cat((det_out_up, det_double_up), dim=1)
            det_out_down = torch.cat(
                (det_out_down, det_double_down), dim=1)

        return det_out_up, det_out_down

    def operator(self, mo, bop, op=op.add, op_squared=False):
        """Computes the values of an opearator applied to the procuts of determinant

        Args:
            mo (torch.tensor): matrix of MO vals(Nbatch, Nelec, Nmo)
            bkin (torch.tensor): kinetic operator (Nbatch, Nelec, Nmo)
            op (operator): how to combine the up/down contribution
            op_squared (bool, optional) return the trace of the square of the product if True

        Returns:
            torch.tensor: kinetic energy
        """

        # get the values of the operator
        if self.config_method == 'ground_state':
            op_vals = self.operator_ground_state(mo, bop, op_squared)

        elif self.config_method.startswith('single'):
            op_vals = self.operator_single_double(mo, bop, op_squared)

        elif self.config_method.startswith('cas('):
            op_vals = self.operator_explicit(mo, bop, op_squared)

        else:
            raise ValueError(
                'Configuration %s not recognized' % self.config_method)

        # combine the values is necessary
        if op is not None:
            return op(*op_vals)
        else:
            return op_vals

    def operator_ground_state(self, mo, bop, op_squared=False):
        """Computes the values of any operator on gs only

        Args:
            mo (torch.tensor): matrix of molecular orbitals
            bkin (torch.tensor): matrix of kinetic operator
            op_squared (bool, optional) return the trace of the square of the product if True

        Returns:
            torch.tensor: operator values
        """

        # occupied orbital matrix + det and inv on spin up
        Aocc_up = mo[:, :self.nup, :self.nup]

        # occupied orbital matrix + det and inv on spin down
        Aocc_down = mo[:, self.nup:, :self.ndown]

        # inverse of the
        invAup = torch.inverse(Aocc_up)
        invAdown = torch.inverse(Aocc_down)

        # precompute the product A^{-1} B
        op_ground_up = invAup @ bop[..., :self.nup, :self.nup]
        op_ground_down = invAdown @ bop[..., self.nup:, :self.ndown]

        if op_squared:
            op_ground_up = op_ground_up @ op_ground_up
            op_ground_down = op_ground_down @ op_ground_down

        # ground state operator
        op_ground_up = btrace(op_ground_up)
        op_ground_down = btrace(op_ground_down)

        op_ground_up.unsqueeze_(-1)
        op_ground_down.unsqueeze_(-1)

        return op_ground_up, op_ground_down

    def operator_explicit(self, mo, bkin, op_squared=False):
        r"""Computes the value of any operator using the trace trick for a product
            of spin up/down determinant.

        .. math::
            -\\frac{1}{2} \Delta \Psi = -\\frac{1}{2}  D_{up} D_{down}
            ( \Delta_{up} D_{up} / D_{up} + \Delta_{down} D_{down}  / D_{down} )

        Args:
            mo (torch.tensor): matrix of MO vals(Nbatch, Nelec, Nmo)
            bkin (torch.tensor): kinetic operator (Nbatch, Nelec, Nmo)
            op_squared (bool, optional) return the trace of the square of the product if True

        Returns:
            torch.tensor: kinetic energy
        """

        # shortcut up/down matrices
        Aup, Adown = self.orb_proj.split_orbitals(mo)
        Bup, Bdown = self.orb_proj.split_orbitals(bkin)

        # check ifwe have 1 or multiple ops
        multiple_op = (Bup.ndim == 5)

        # inverse of MO matrices
        iAup = torch.inverse(Aup)
        iAdown = torch.inverse(Adown)

        # if we have multiple operators
        if multiple_op:
            iAup = iAup.unsqueeze(1)
            iAdown = iAdown.unsqueeze(1)

        # precompute product invA x B
        op_val_up = iAup @ Bup
        op_val_down = iAdown @ Bdown

        if op_squared:
            op_val_up = op_val_up @ op_val_up
            op_val_down = op_val_down @ op_val_down

        # kinetic terms
        op_val_up = btrace(op_val_up)
        op_val_down = btrace(op_val_down)

        # reshape
        if multiple_op:
            op_val_up = op_val_up.permute(1, 2, 0)
            op_val_down = op_val_down.permute(1, 2, 0)
        else:
            op_val_up = op_val_up.transpose(0, 1)
            op_val_down = op_val_down.transpose(0, 1)

        return (op_val_up, op_val_down)

    def operator_single_double(self, mo, bop, op_squared=False):
        """Computes the value of any operator on gs + single + double

        Args:
            mo (torch.tensor): matrix of molecular orbitals
            bkin (torch.tensor): matrix of kinetic operator
            op_squared (bool, optional) return the trace of the square of the product if True

        Returns:
            torch.tensor: kinetic energy values
        """

        op_up, op_down = self.operator_unique_single_double(
            mo, bop, op_squared)

        return (op_up[..., self.index_unique_excitation[0]],
                op_down[..., self.index_unique_excitation[1]])

    def operator_unique_single_double(self, mo, bop, op_squared):
        """Compute the operator value of the unique single/double conformation

        Args:
            mo ([type]): [description]
            bkin ([type]): [description]
            op_squared (bool) return the trace of the square of the product
        """

        nbatch = mo.shape[0]

        if not hasattr(self.exc_mask, 'index_unique_single_up'):
            self.exc_mask.get_index_unique_single()

        if not hasattr(self.exc_mask, 'index_unique_double_up'):
            self.exc_mask.get_index_unique_double()

        do_single = len(self.exc_mask.index_unique_single_up) != 0
        do_double = len(self.exc_mask.index_unique_double_up) != 0

        # occupied orbital matrix + det and inv on spin up
        Aocc_up = mo[:, :self.nup, :self.nup]

        # occupied orbital matrix + det and inv on spin down
        Aocc_down = mo[:, self.nup:, :self.ndown]

        # inverse of the
        invAup = torch.inverse(Aocc_up)
        invAdown = torch.inverse(Aocc_down)

        # precompute invA @ B
        invAB_up = invAup @ bop[..., :self.nup, :self.nup]
        invAB_down = invAdown @ bop[..., self.nup:, :self.ndown]

        # ground state operator
        if op_squared:
            op_ground_up = btrace(invAB_up@invAB_up)
            op_ground_down = btrace(invAB_down@invAB_down)
        else:
            op_ground_up = btrace(invAB_up)
            op_ground_down = btrace(invAB_down)

        op_ground_up.unsqueeze_(-1)
        op_ground_down.unsqueeze_(-1)

        # store the kin terms we need
        op_out_up = op_ground_up.clone()
        op_out_down = op_ground_down.clone()

        # virtual orbital matrices spin up/down
        Avirt_up = mo[:, :self.nup, self.nup:self.index_max_orb_up]
        Avirt_down = mo[:, self.nup:,
                        self.ndown: self.index_max_orb_down]

        # compute the products of invA and Btilde
        mat_exc_up = (invAup @ Avirt_up)
        mat_exc_down = (invAdown @ Avirt_down)

        bop_up = bop[..., :self.nup, :self.index_max_orb_up]
        bop_occ_up = bop[..., :self.nup, :self.nup]
        bop_virt_up = bop[..., :self.nup,
                          self.nup:self.index_max_orb_up]

        bop_down = bop[:, self.nup:, :self.index_max_orb_down]
        bop_occ_down = bop[..., self.nup:, :self.ndown]
        bop_virt_down = bop[..., self.nup:,
                            self.ndown:self.index_max_orb_down]

        Mup = invAup @ bop_virt_up - invAup @ bop_occ_up @ invAup @ Avirt_up
        Mdown = invAdown @ bop_virt_down - \
            invAdown @ bop_occ_down @ invAdown @ Avirt_down

        # if we only want the normal value of the op and not its squared
        if not op_squared:

            # reshape the M matrices
            Mup = Mup.view(*Mup.shape[:-2], -1)
            Mdown = Mdown.view(*Mdown.shape[:-2], -1)

            if do_single:

                # spin up
                op_sin_up = self.op_single(op_ground_up, mat_exc_up, Mup,
                                           self.exc_mask.index_unique_single_up, nbatch)

                # spin down
                op_sin_down = self.op_single(op_ground_down, mat_exc_down, Mdown,
                                             self.exc_mask.index_unique_single_down, nbatch)

                # store the terms we need
                op_out_up = torch.cat((op_out_up, op_sin_up), dim=-1)
                op_out_down = torch.cat(
                    (op_out_down, op_sin_down), dim=-1)

            if do_double:

                # spin up
                op_dbl_up = self.op_multiexcitation(op_ground_up, mat_exc_up, Mup,
                                                    self.exc_mask.index_unique_double_up,
                                                    2, nbatch)

                # spin down
                op_dbl_down = self.op_multiexcitation(op_ground_down, mat_exc_down, Mdown,
                                                      self.exc_mask.index_unique_double_down,
                                                      2, nbatch)

                # store the terms we need
                op_out_up = torch.cat((op_out_up, op_dbl_up), dim=-1)
                op_out_down = torch.cat(
                    (op_out_down, op_dbl_down), dim=-1)

            return op_out_up, op_out_down

        # if we watn the squre of the operatore
        # typically trace(ABAB)
        else:

            # compute A^-1 B M
            Yup = invAB_up @ Mup
            Ydown = invAB_down @ Mdown

            # reshape the M matrices
            Mup = Mup.view(*Mup.shape[:-2], -1)
            Mdown = Mdown.view(*Mdown.shape[:-2], -1)

            # reshape the Y matrices
            Yup = Yup.view(*Yup.shape[:-2], -1)
            Ydown = Ydown.view(*Ydown.shape[:-2], -1)

            if do_single:

                # spin up
                op_sin_up = self.op_squared_single(op_ground_up, mat_exc_up,
                                                   Mup, Yup,
                                                   self.exc_mask.index_unique_single_up,
                                                   nbatch)

                # spin down
                op_sin_down = self.op_squared_single(op_ground_down, mat_exc_down,
                                                     Mdown, Ydown,
                                                     self.exc_mask.index_unique_single_down,
                                                     nbatch)

                # store the terms we need
                op_out_up = torch.cat((op_out_up, op_sin_up), dim=-1)
                op_out_down = torch.cat(
                    (op_out_down, op_sin_down), dim=-1)

            if do_double:

                # spin up values
                op_dbl_up = self.op_squared_multiexcitation(op_ground_up, mat_exc_up,
                                                            Mup, Yup,
                                                            self.exc_mask.index_unique_double_down,
                                                            2, nbatch)

                # spin down values
                op_dbl_down = self.op_squared_multiexcitation(op_ground_down, mat_exc_down,
                                                              Mdown, Ydown,
                                                              self.exc_mask.index_unique_double_down,
                                                              2, nbatch)

                # store the terms we need
                op_out_up = torch.cat((op_out_up, op_dbl_up), dim=-1)
                op_out_down = torch.cat(
                    (op_out_down, op_dbl_down), dim=-1)

            return op_out_up, op_out_down

    @staticmethod
    def op_single(baseterm, mat_exc, M, index, nbatch):
        r"""Computes the operator values for single excitation

        .. math::
            Tr( \bar{A}^{-1} \bar{B}) = Tr(A^{-1} B) + Tr( T M )
            T = P ( A^{-1} \bar{A})^{-1} P
            M = A^{-1}\bar{B} - A^{-1}BA^{-1}\bar{A}

        Args:
            baseterm (torch.tensor): trace(A B)
            mat_exc (torch.tensor): invA @ Abar
            M (torch.tensor): invA Bbar - inv A B inv A Abar
            index(List): list of index of the excitations
            nbatch : batch size
        """

        # compute the values of T
        T = (1. / mat_exc.view(nbatch, -1)[:, index])

        # computes trace(T M)
        op_vals = T * M[..., index]

        # add the base terms
        op_vals += baseterm

        return op_vals

    @staticmethod
    def op_multiexcitation(baseterm, mat_exc, M, index, size, nbatch):
        r"""Computes the operator values for single excitation

        .. math::
            Tr( \bar{A}^{-1} \bar{B}) = Tr(A^{-1} B) + Tr( T M )
            T = P ( A^{-1} \bar{A})^{-1} P
            M = A^{-1}\bar{B} - A^{-1}BA^{-1}\bar{A}

        Args:
            baseterm (torch.tensor): trace(A B)
            mat_exc (torch.tensor): invA @ Abar
            M (torch.tensor): invA Bbar - inv A B inv A Abar
            index(List): list of index of the excitations
            size(int) : number of excitation
            nbatch : batch size
        """

        # get the values of the excitation matrix invA Abar
        T = mat_exc.view(nbatch, -1)[:, index]

        # get the shapes of the size x size matrices
        _ext_shape = (*T.shape[:-1], -1, size, size)
        _m_shape = (*M.shape[:-1], -1, size, size)

        # computes the inverse of invA Abar
        T = torch.inverse(T.view(_ext_shape))

        # computes T @ M (after reshaping M as size x size matrices)
        op_vals = T @ (M[..., index]).view(_m_shape)

        # compute the trace
        op_vals = btrace(op_vals)

        # add the base term
        op_vals += baseterm

        return op_vals

    @staticmethod
    def op_squared_single(baseterm, mat_exc, M, Y, index, nbatch):
        r"""Computes the operator squared for single excitation

        .. math::
            Tr( (\bar{A}^{-1} \bar{B})^2) = Tr((A^{-1} B)^2) + Tr( (T M)^2 ) + 2 Tr(T Y)
            T = P ( A^{-1} \bar{A})^{-1} P -> mat_exc in the code
            M = A^{-1}\bar{B} - A^{-1}BA^{-1}\bar{A}
            Y = A^{-1} B M

        Args:
            baseterm (torch.tensor): trace(A B A B)
            mat_exc (torch.tensor): invA @ Abar
            M (torch.tensor): invA Bbar - inv A B inv A Abar
            Y (torch.tensor): invA B M
            index(List): list of index of the excitations
            nbatch : batch size
        """

        # get the values of the inverse excitation matrix
        T = 1. / (mat_exc.view(nbatch, -1)[:, index])

        # compute  trace(( T M )^2)
        tmp = (T * M[..., index])
        op_vals = tmp*tmp

        # trace(T Y)
        tmp = (T * Y[..., index])
        op_vals += 2 * tmp

        # add the base term
        op_vals += baseterm

        return op_vals

    @staticmethod
    def op_squared_multiexcitation(baseterm, mat_exc, M, Y, index, size, nbatch):
        r"""Computes the operator squared for multiple excitation

        .. math::
            Tr( (\bar{A}^{-1} \bar{B})^2) = Tr((A^{-1} B)^2) + Tr( (T M)^2 ) + 2 Tr(T Y)
            T = P ( A^{-1} \bar{A})^{-1} P -> mat_exc in the code
            M = A^{-1}\bar{B} - A^{-1}BA^{-1}\bar{A}
            Y = A^{-1} B M

        Args:
            baseterm (torch.tensor): trace(A B A B)
            mat_exc (torch.tensor): invA @ Abar
            M (torch.tensor): invA Bbar - inv A B inv A Abar
            Y (torch.tensor): invA B M
            index(List): list of index of the excitations
            nbatch : batch size
            size(int): number of excitation
        """

        # get the values of the excitation matrix invA Abar
        T = mat_exc.view(nbatch, -1)[:, index]

        # get the shape as a series of size x size matrices
        _ext_shape = (*T.shape[:-1], -1, size, size)
        _m_shape = (*M.shape[:-1], -1, size, size)
        _y_shape = (*Y.shape[:-1], -1, size, size)

        # reshape T and take the inverse of the matrices
        T = torch.inverse(T.view(_ext_shape))

        # compute  trace(( T M )^2)
        tmp = T @ (M[..., index]).view(_m_shape)

        # take the trace of that and add to base value
        tmp = btrace(tmp @ tmp)
        op_vals = tmp

        # compute trace( T Y )
        tmp = T @ (Y[..., index]).view(_y_shape)
        tmp = btrace(tmp)
        op_vals += 2*tmp

        # add the base term
        op_vals += baseterm

        return op_vals
