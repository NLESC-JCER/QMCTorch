import torch
import numpy as np


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


class OrbitalMask(object):

    def __init__(self, configs, mol):
        """Select the occupied MOs of Slater determinant using masks

        Args:
            configs (list): configurations of the slater determinants
            mol (Molecule): Molecule object
        """

        self.configs = configs
        self.nconfs = len(configs[0])
        self.nmo = mol.basis.nmo
        self.nup = mol.nup
        self.ndown = mol.ndown
        self.nelec = mol.nelec

        self.masks = self.get_masks()

    def get_masks(self):
        """Get the Boolean mask to extract the orbitals."""

        mask_up = torch.zeros(
            self.nconfs, self.nelec, self.nmo).type(torch.bool)
        mask_down = torch.zeros(
            self.nconfs, self.nelec, self.nmo).type(torch.bool)

        for ic, (cup, cdown) in enumerate(
                zip(self.configs[0], self.configs[1])):

            mask_up[ic][np.ix_(range(self.nup), cup)] = True
            mask_down[ic][np.ix_(
                range(self.nup, self.nelec), cdown)] = True

        return torch.cat(mask_up, mask_down)

    def split_orbitals(self, mo):
        """Split the orbital  matrix in multiple slater matrices

        Args:
            mo (torch.tensor): molecular orbital matrix

        Returns:
            torch.tensor: all slater matrices
        """
        nbatch = mo.shape[0]
        len_up = nbatch * self.nup * self.nup
        _tmp = mo.masked_select(self.masks)

        return _tmp[:len_up].view(nbatch, self.nup, self.nup), \
            _tmp[len_up:].view(nbatch, self.ndown, self.ndown)


class ExcitationMask(object):

    def __init__(self, excitations, mol):
        """Select the occupied MOs of Slater determinant using masks

        Args:
            configs (list): configurations of the slater determinants
            mol (Molecule): Molecule object
        """

        self.excitations = excitations
        self.nconfs = len(excitations[0])
        self.nmo = mol.basis.nmo
        self.nup = mol.nup
        self.ndown = mol.ndown
        self.nelec = mol.nelec

        self.mask_up, self.mask_down = self.get_masks()

    def get_masks(self):
        """Get the masks associated with the excitations."""

        mask_up = torch.zeros(
            self.nconfs, self.nup, self.nmo).type(torch.bool)
        mask_down = torch.zeros(
            self.nconfs, self.ndown, self.nmo).type(torch.bool)

        for ix, (exc_up, exc_down) in enumerate(
                zip(self.excitations[0], self.excitations[1])):

            mask_up[ix][np.ix_(exc_up[0], exc_up[1])] = True
            mask_down[ix][np.ix_(exc_down[0], exc_down[1])] = True

        return mask_up, mask_down

    def compute_determinant(self, input_up, input_down):
        """computes the determinant of all the matrices defined by the mask

        Args:
            input_up (torch.tensor): typically matrix of A^{-1} B fo spin up
            input_down (torch.tensor): typically matrix of A^{-1} B fo spin down

        Returns:
            torch.tensor : values of the determinants
        """

        xcup = input_up.masked_select(
            self.mask_up).split(self.index_up)
        xcdown = input_down.masked_select(
            self.mask_down).split(self.index_down)
