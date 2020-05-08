
import torch
from torch import nn
from .orbital_projector import OrbitalProjector, ExcitationMask


class SlaterPooling(nn.Module):

    """Applies a slater determinant pooling in the active space."""

    def __init__(self, configs, mol, cuda=False):
        """Computes the Sater determinants

        Args:
            configs (tuple): configuratin of the electrons
            mol (Molecule): Molecule instance
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.

        """
        super(SlaterPooling, self).__init__()

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

        det_unique_up, det_unique_down = self.det_unique_single(input)
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

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo
        """

        nbatch = input.shape[0]

        if not hasattr(self.exc_mask, 'mask_unique_single_up'):
            self.exc_mask.get_mask_unique_single()

        Aup = input[:, :self.nup, :self.nup]
        Adown = input[:, self.nup:, :self.ndown]

        detAup = torch.det(Aup)
        invAup = torch.inverse(Aup)

        detAdown = torch.det(Adown)
        invAdown = torch.inverse(Adown)

        Bup = input[:, :self.nup, self.nup:self.index_max_orb_up]
        Bdown = input[:, self.nup:,
                      self.ndown: self.index_max_orb_down]

        det_up = (invAup @ Bup).view(
            nbatch, -1)[:, self.exc_mask.index_unique_single_up]

        det_down = (invAdown @ Bdown).view(
            nbatch, -1)[:, self.exc_mask.index_unique_single_down]

        det_up = detAup.unsqueeze(-1) * det_up.view(nbatch, -1)
        det_up *= self.exc_mask.sign_unique_single_up

        det_down = detAdown.unsqueeze(-1) * det_down.view(nbatch, -1)
        det_down *= self.exc_mask.sign_unique_single_down

        return torch.cat((detAup.unsqueeze(-1), det_up), dim=1),\
            torch.cat((detAdown.unsqueeze(-1), det_down), dim=1)

    def det_excitation(self, input):
        """Computes the values of the determinants from the slater matrices

        Args:
            input (torch.tensor): MO matrices nbatch x nelec x nmo

        Returns:
            torch.tensor: slater determinants
        """
        detAup = torch.det(input[:, self.nup, :self.nup])
        invAup = torch.inverse(input[:, self.nup, :self.nup])
        xcup = (invAup @ input[:, :self.nup, :]
                ).masked_select(self.exc_mask.mask_up)
