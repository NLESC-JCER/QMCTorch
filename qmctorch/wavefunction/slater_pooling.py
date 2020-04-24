
import torch
from torch import nn
from .orbital_projector import OrbitalProjector


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

        self.configs = configs
        self.nconfs = len(configs[0])

        self.nmo = mol.basis.nmo
        self.nup = mol.nup
        self.ndown = mol.ndown

        self.orb_proj = OrbitalProjector(configs, mol)

        if cuda:
            self.device = torch.device('cuda')
            self.orb_proj.Pup = self.orb_proj.Pup.to(self.device)
            self.orb_proj.Pdown = self.orb_proj.Pdown.to(self.device)

    def forward(self, input, return_matrix=False):
        """Computes the values of the determinats

        Args:
            input (torch.tensor): MO matrices nbatc x nelec x nmo
            return_matrix (bool, optional): if true return the slater matrices 
                                            instead of their determinats. 
                                            Defaults to False.

        Returns:
            torch.tensor: slater determinants
        """

        mo_up, mo_down = self.orb_proj.split_orbitals(input)
        if return_matrix:
            return mo_up, mo_down
        else:
            return (
                torch.det(mo_up) *
                torch.det(mo_down)).transpose(0, 1)
