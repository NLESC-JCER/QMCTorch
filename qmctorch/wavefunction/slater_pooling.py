
import torch
from torch import nn
from .orbital_configurations import get_excitation, get_unique_excitation
from .orbital_projector import OrbitalProjector, ExcitationMask
from ..utils import btrace, bdet2


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

            # if the orbital in configs are in increasing order
            # we should deal with that better ...
            # det_up *= self.exc_mask.sign_unique_single_up
            # det_down *= self.exc_mask.sign_unique_single_down

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

    def kinetic(self, mo, bkin):
        """Computes the values of the kinetic energy

        Args:
            mo (torch.tensor): matrix of MO vals(Nbatch, Nelec, Nmo)
            bkin (torch.tensor): kinetic operator (Nbatch, Nelec, Nmo)

        Returns:
            torch.tensor: kinetic energy
        """

        if self.config_method.startswith('cas('):
            return self.kinetic_explicit(mo, bkin)
        else:
            return self.kinetic_single_double(mo, bkin)

    def kinetic_explicit(self, mo, bkin):
        r"""Compute the kinetic energy using the trace trick for a product of spin up/down determinant.

        .. math::
            -\\frac{1}{2} \Delta \Psi = -\\frac{1}{2}  D_{up} D_{down}
            ( \Delta_{up} D_{up}  + \Delta_{down} D_{down} )

        Args:
            mo (torch.tensor): matrix of MO vals(Nbatch, Nelec, Nmo)
            bkin (torch.tensor): kinetic operator (Nbatch, Nelec, Nmo)

        Returns:
            torch.tensor: kinetic energy
        """

        # shortcut up/down matrices
        Aup, Adown = self.orb_proj.split_orbitals(mo)
        Bup, Bdown = self.orb_proj.split_orbitals(bkin)

        # inverse of MO matrices
        iAup = torch.inverse(Aup)
        iAdown = torch.inverse(Adown)

        # determinant product
        det_prod = torch.det(Aup) * torch.det(Adown)

        # kinetic terms
        kinetic = (btrace(iAup@Bup) + btrace(iAdown@Bdown))

        # reshape
        kinetic = kinetic.transpose(0, 1)

        return kinetic

    def kinetic_single_double(self, mo, bkin):
        """Computes the kinetic energy of gs + single + double

        Args:
            mo (torch.tensor): matrix of molecular orbitals
            bkin (torch.tensor): matrix of kinetic operator

        Returns:
            torch.tensor: kinetic energy values
        """

        kin_up, kin_down = self.operator_unique_single_double(
            mo, bkin)

        return (kin_up[:, self.index_unique_excitation[0]] +
                kin_down[:, self.index_unique_excitation[1]])

    def operator_unique_single_double(self, mo, bop):
        """Compute the kinetic energy of the unique single/double conformation

        Args:
            mo ([type]): [description]
            bkin ([type]): [description]
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
        detAup = torch.det(Aocc_up)

        # occupied orbital matrix + det and inv on spin down
        Aocc_down = mo[:, self.nup:, :self.ndown]
        detAdown = torch.det(Aocc_down)

        # inverse of the
        invAup = torch.inverse(Aocc_up)
        invAdown = torch.inverse(Aocc_down)

        # ground state operator
        op_ground_up = btrace(
            invAup @ bop[..., :self.nup, :self.nup])
        op_ground_down = btrace(
            invAdown @ bop[..., self.nup:, :self.ndown])

        op_ground_up.unsqueeze_(-1)
        op_ground_down.unsqueeze_(-1)

        # store the kin terms we need
        op_out_up = op_ground_up.clone()
        op_out_down = op_ground_down.clone()

        if self.config_method == 'ground_state':
            return op_out_up, op_out_down

        # virtual orbital matrices spin up/down
        Avirt_up = mo[:, :self.nup, self.nup:self.index_max_orb_up]
        Avirt_down = mo[:, self.nup:,
                        self.ndown: self.index_max_orb_down]

        # compute the products of Ain and B
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

        # reshape the M matrices
        Mup = Mup.view(*Mup.shape[:-2], -1)
        Mdown = Mdown.view(*Mdown.shape[:-2], -1)

        if do_single:

            op_sin_up = (1. / mat_exc_up.view(nbatch, -1)[:, self.exc_mask.index_unique_single_up]) * \
                Mup[..., self.exc_mask.index_unique_single_up]
            op_sin_up += op_ground_up

            op_sin_down = (1. / mat_exc_down.view(nbatch, -1)[:, self.exc_mask.index_unique_single_down]) * \
                Mdown[..., self.exc_mask.index_unique_single_down]
            op_sin_down += op_ground_down

            # store the terms we need
            op_out_up = torch.cat((op_out_up, op_sin_up), dim=-1)
            op_out_down = torch.cat(
                (op_out_down, op_sin_down), dim=-1)

        if do_double:

            op_dbl_up = mat_exc_up.view(
                nbatch, -1)[:, self.exc_mask.index_unique_double_up]

            _shape = (*op_dbl_up.shape[:-1], -1, 2, 2)

            op_dbl_up = torch.inverse(op_dbl_up.view(_shape))
            print(op_dbl_up.shape)
            print(Mup[..., self.exc_mask.index_unique_double_up].shape)
            op_dbl_up = op_dbl_up @ (
                Mup[..., self.exc_mask.index_unique_double_up]).view(_shape)
            op_dbl_up = btrace(op_dbl_up)
            op_dbl_up += op_ground_up

            op_dbl_down = mat_exc_down.view(
                nbatch, -1)[:, self.exc_mask.index_unique_double_down]
            op_dbl_down = torch.inverse(
                op_dbl_down.view(_shape))
            op_dbl_down = op_dbl_down @ (
                Mdown[..., self.exc_mask.index_unique_double_down]).view(_shape)
            op_dbl_down = btrace(op_dbl_down)
            op_dbl_down += op_ground_down

            # store the terms we need
            op_out_up = torch.cat((op_out_up, op_dbl_up), dim=-1)
            op_out_down = torch.cat(
                (op_out_down, op_dbl_down), dim=-1)

        return op_out_up, op_out_down
