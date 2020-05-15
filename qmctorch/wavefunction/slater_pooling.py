
import torch
from torch import nn
from .orbital_projector import OrbitalProjector, ExcitationMask


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
        self.process_configs(configs)

        self.nmo = mol.basis.nmo
        self.nup = mol.nup
        self.ndown = mol.ndown

        self.orb_proj = OrbitalProjector(configs, mol)
        self.exc_mask = ExcitationMask(self.unique_excitation, mol,
                                       (self.index_max_orb_up, self.index_max_orb_down))

        if cuda:
            self.device = torch.device('cuda')
            self.orb_proj.Pup = self.orb_proj.Pup.to(self.device)
            self.orb_proj.Pdown = self.orb_proj.Pdown.to(self.device)

    def process_configs(self, configs):
        """Extract all necessary info from configs.

        Args:
            configs (tuple): configuratin of the electrons

        """

        self.configs = configs
        self.nconfs = len(configs[0])
        self.index_max_orb_up = self.configs[0].max().item() + 1
        self.index_max_orb_down = self.configs[1].max().item() + 1

        self.excitation_index = self.get_excitation(configs)
        self.unique_excitation, self.index_unique_excitation = self.get_unique_excitation(
            configs)

    def get_excitation(self, configs):
        """get the excitation data

        Args:
            configs (tuple): configuratin of the electrons

        Returns:
            exc_up, exc_down : index of the obitals in the excitaitons
                               [i,j],[l,m] : excitation i -> l, j -> l
        """
        exc_up, exc_down = [], []
        for ic, (cup, cdown) in enumerate(zip(configs[0], configs[1])):

            set_cup = set(tuple(cup.tolist()))
            set_cdown = set(tuple(cdown.tolist()))

            if ic == 0:
                set_gs_up = set_cup
                set_gs_down = set_cdown

            else:
                exc_up.append([list(set_gs_up.difference(set_cup)),
                               list(set_cup.difference(set_gs_up))])

                exc_down.append([list(set_gs_down.difference(set_cdown)),
                                 list(set_cdown.difference(set_gs_down))])

        return (exc_up, exc_down)

    def get_unique_excitation(self, configs):
        """get the unique excitation data

        Args:
            configs (tuple): configuratin of the electrons

        Returns:
            exc_up, exc_down : index of the obitals in the excitaitons
                               [i,j],[l,m] : excitation i -> l, j -> l
            index_up, index_down : index map for the unique exc
                                 [0,0,...], [0,1,...] means that 
                                 1st : excitation is composed of unique_up[0]*unique_down[0]
                                 2nd : excitation is composed of unique_up[0]*unique_down[1]
                                 ....

        """
        uniq_exc_up, uniq_exc_down = [], []
        index_uniq_exc_up, index_uniq_exc_down = [], []
        for ic, (cup, cdown) in enumerate(zip(configs[0], configs[1])):

            set_cup = set(tuple(cup.tolist()))
            set_cdown = set(tuple(cdown.tolist()))

            if ic == 0:
                set_gs_up = set_cup
                set_gs_down = set_cdown

            exc_up = [list(set_gs_up.difference(set_cup)),
                      list(set_cup.difference(set_gs_up))]

            exc_down = [list(set_gs_down.difference(set_cdown)),
                        list(set_cdown.difference(set_gs_down))]

            if exc_up not in uniq_exc_up:
                uniq_exc_up.append(exc_up)

            if exc_down not in uniq_exc_down:
                uniq_exc_down.append(exc_down)

            index_uniq_exc_up.append(uniq_exc_up.index(exc_up))
            index_uniq_exc_down.append(
                uniq_exc_down.index(exc_down))

        return (uniq_exc_up, uniq_exc_down), (index_uniq_exc_up, index_uniq_exc_down)

    def forward(self, input):
        """Computes the values of the determinats

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo

        Returns:
            torch.tensor: slater determinants
        """
        if self.config_method.startswith('single'):
            return self.det_single(input)
        else:
            return self.det_explicit(input)

    def det_explicit(self, input):
        """Computes the values of the determinants from the slater matrices

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo

        Returns:
            torch.tensor: slater determinants
        """

        mo_up, mo_down = self.get_slater_matrices(input)
        return (torch.det(mo_up) * torch.det(mo_down)).transpose(0, 1)

    def det_single(self, input):
        """Computes the determinant of ground state + single

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo

        Returns:
            torch.tensor: slater determinants
        """

        # compute the determinant of the unique single excitation
        det_unique_up, det_unique_down = self.det_unique_single(input)

        # returns the product of spin up/down required by each excitation
        return (det_unique_up[:, self.index_unique_excitation[0]] *
                det_unique_down[:, self.index_unique_excitation[1]])

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

    def get_slater_matrices(self, input):
        """Computes the slater matrices

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo


        Returns:
            (torch.tensor, torch.tensor): slater matrices of spin up/down
        """
        return self.orb_proj.split_orbitals(input)

    def det_ground_state(self, input):
        """Computes the SD of the ground state

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo
        """

        if not hasattr(self.exc_mask, 'mask_unique_single_up'):
            self.exc_mask.get_mask_unique_single()

        return (torch.det(input[:, :self.nup, :self.nup]),
                torch.det(input[:, self.nup:, :self.ndown]))

    def det_unique_single(self, input):
        """Computes the SD of single excitations

        .. note:: The determinants of the single excitations 
        are calculated from the ground state determinant and 
        the ground state Slater matrices whith one column modified.
        See : Monte Carlo Methods in ab initio quantum chemistry 
              B.L. Hammond, appendix B1

        Note ; if the state on coonfigs are specified in order 
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

        # occupied orbital matrix + det and inv on spin up
        Aup = input[:, :self.nup, :self.nup]
        detAup = torch.det(Aup)
        invAup = torch.inverse(Aup)

        # occupied orbital matrix + det and inv on spin down
        Adown = input[:, self.nup:, :self.ndown]
        detAdown = torch.det(Adown)
        invAdown = torch.inverse(Adown)

        # virtual orbital matrices spin up/down
        Bup = input[:, :self.nup, self.nup:self.index_max_orb_up]
        Bdown = input[:, self.nup:,
                      self.ndown: self.index_max_orb_down]

        # determinant of the unique excitation spin up
        det_up = (invAup @ Bup).view(
            nbatch, -1)[:, self.exc_mask.index_unique_single_up]

        # determinant of the unique excitation spin down
        det_down = (invAdown @ Bdown).view(
            nbatch, -1)[:, self.exc_mask.index_unique_single_down]

        # multiply with ground state determinant
        # and account for permutation for deep excitation
        det_up = detAup.unsqueeze(-1) * det_up.view(nbatch, -1)

        # multiply with ground state determinant
        # and account for permutation for deep excitation
        det_down = detAdown.unsqueeze(-1) * det_down.view(nbatch, -1)

        # if the orbital in configs are in increasing order
        # we should deal with that better ...
        # det_up *= self.exc_mask.sign_unique_single_up
        # det_down *= self.exc_mask.sign_unique_single_down

        return torch.cat((detAup.unsqueeze(-1), det_up), dim=1),\
            torch.cat((detAdown.unsqueeze(-1), det_down), dim=1)

    # def det_excitation(self, input):
    #     """Computes the values of the determinants from the slater matrices

    #     Args:
    #         input (torch.tensor): MO matrices nbatch x nelec x nmo

    #     Returns:
    #         torch.tensor: slater determinants
    #     """
    #     detAup = torch.det(input[:, self.nup, :self.nup])
    #     invAup = torch.inverse(input[:, self.nup, :self.nup])
    #     xcup = (invAup @ input[:, :self.nup, :]
    #             ).masked_select(self.exc_mask.mask_up)

    def det_unique_single_double(self, input):
        """Computes the SD of single/double excitations

        .. note:: The determinants of the single excitations 
        are calculated from the ground state determinant and 
        the ground state Slater matrices whith one column modified.
        See : Monte Carlo Methods in ab initio quantum chemistry 
              B.L. Hammond, appendix B1

        Note ; if the state on coonfigs are specified in order 
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

        # occupied orbital matrix + det and inv on spin up
        Aup = input[:, :self.nup, :self.nup]
        detAup = torch.det(Aup)
        invAup = torch.inverse(Aup)

        # occupied orbital matrix + det and inv on spin down
        Adown = input[:, self.nup:, :self.ndown]
        detAdown = torch.det(Adown)
        invAdown = torch.inverse(Adown)

        # virtual orbital matrices spin up/down
        Bup = input[:, :self.nup, self.nup:self.index_max_orb_up]
        Bdown = input[:, self.nup:,
                      self.ndown: self.index_max_orb_down]

        # compute the products of Ain and B
        mat_exc_up = (invAup @ Bup)
        mat_exc_down = (invAdown @ Bdown)

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

        det_double_up = mat_exc_up.view(
            nbatch, -1)[:, self.exc_mask.index_unique_double_up]

        det_double_up = torch.det(
            det_double_up.view(nbatch, -1, 2, 2))

        det_double_up = detAup.unsqueeze(-1) * det_double_up

        det_double_down = mat_exc_down.view(
            nbatch, -1)[:, self.exc_mask.index_unique_double_down]

        det_double_down = torch.det(
            det_double_down.view(nbatch, -1, 2, 2))

        det_double_down = detAdown.unsqueeze(-1) * det_double_down

        return torch.cat((detAup.unsqueeze(-1), det_single_up, det_double_up), dim=1),\
            torch.cat((detAdown.unsqueeze(-1),
                       det_single_down, det_double_down), dim=1)
