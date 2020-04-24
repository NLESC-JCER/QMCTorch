import torch


class OrbitalProjector(object):

    def __init__(self, configs, mol):
        """Project the MO matrix in Slater Matrices

        Args:
            configs (list): configurations of the slater determinants
            mol (Molecule): Molecule object
        """

        self.configs = configs
        self.nconfs = len(configs[0])
        self.nmo = mol.basis.nmo
        self.nup = mol.nup
        self.ndown = mol.ndown

        self.Pup, self.Pdown = self.get_projectors()

    def get_projectors(self):
        """Get the projectors of the conf in the CI expansion

        Returns:
            torch.tensor, torch.tensor : projectors
        """

        Pup = torch.zeros(self.nconfs, self.nmo, self.nup)
        Pdown = torch.zeros(self.nconfs, self.nmo, self.ndown)

        for ic, (cup, cdown) in enumerate(
                zip(self.configs[0], self.configs[1])):

            for _id, imo in enumerate(cup):
                Pup[ic][imo, _id] = 1.

            for _id, imo in enumerate(cdown):
                Pdown[ic][imo, _id] = 1.

        return Pup.unsqueeze(1), Pdown.unsqueeze(1)

    def split_orbitals(self, mo):
        """Split the orbital  matrix in multiple slater matrices

        Args:
            mo (torch.tensor): molecular orbital matrix

        Returns:
            torch.tensor: all slater matrices
        """
        return mo[:, :self.nup, :] @ self.Pup, mo[:,
                                                  self.nup:, :] @ self.Pdown
