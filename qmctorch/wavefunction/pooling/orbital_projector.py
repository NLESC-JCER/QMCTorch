import torch


class OrbitalProjector:

    def __init__(self, configs, mol, cuda=False):
        """Project the MO matrix in Slater Matrices

        Args:
            configs (list): configurations of the slater determinants
            mol (Molecule): Molecule object
            cuda (bool): use cuda or not
        """

        self.configs = configs
        self.nconfs = len(configs[0])
        self.nmo = mol.basis.nmo
        self.nup = mol.nup
        self.ndown = mol.ndown
        self.device = torch.device('cpu')
        if cuda:
            self.device = torch.device('cuda')

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

        return Pup.unsqueeze(1).to(self.device), Pdown.unsqueeze(1).to(self.device)

    def split_orbitals(self, mat):
        """Split the orbital  matrix in multiple slater matrices

        Args:
            mat (torch.tensor): matrix to split

        Returns:
            torch.tensor: all slater matrices
        """
        if not hasattr(self, 'Pup'):
            self.Pup, self.Pdown = self.get_projectors()

        if mat.ndim == 4:
            # case for multiple operators
            out_up = mat[..., :self.nup, :] @ self.Pup.unsqueeze(1)
            out_down = mat[..., self.nup:,
                           :] @ self.Pdown.unsqueeze(1)

        else:
            # case for single operator
            out_up = mat[..., :self.nup, :] @ self.Pup
            out_down = mat[..., self.nup:, :] @ self.Pdown

        return out_up, out_down


class ExcitationMask:

    def __init__(self, unique_excitations, mol, max_orb, cuda=False):
        """Select the occupied MOs of Slater determinant using masks

        Args:
            unique_excitations (list): the list of unique excitations
            mol (Molecule): Molecule object
            max_orb (list): the max index of each orb for each spin
            cuda (bool): use cuda or not
        """

        self.unique_excitations = unique_excitations
        self.num_unique_exc = len(unique_excitations[0])
        self.nmo = mol.basis.nmo
        self.nup = mol.nup
        self.ndown = mol.ndown
        self.nelec = mol.nelec
        self.max_orb = max_orb

        self.device = torch.device('cpu')
        if cuda:
            self.device = torch.device('cuda')

    def get_index_unique_single(self):
        """Computes the 1D index and permutation
           for the unique singles."""

        ncol_up = self.max_orb[0]-self.nup
        ncol_down = self.max_orb[1]-self.ndown

        self.index_unique_single_up = []
        self.index_unique_single_down = []

        self.sign_unique_single_up = []
        self.sign_unique_single_down = []

        for exc_up, exc_down in zip(self.unique_excitations[0],
                                    self.unique_excitations[1]):

            if len(exc_up[0]) == 1:
                ielec, iorb = exc_up[0][0], exc_up[1][0]
                icol = iorb-self.nup

                self.index_unique_single_up.append(
                    ielec*ncol_up + icol)

                npermut = self.nup-ielec-1
                self.sign_unique_single_up.append((-1)**(npermut))

            if len(exc_down[1]) == 1:
                ielec, iorb = exc_down[0][0], exc_down[1][0]
                icol = iorb-self.ndown

                self.index_unique_single_down.append(
                    ielec*ncol_down + icol)

                npermut = self.ndown-ielec-1
                self.sign_unique_single_down.append((-1)**(npermut))

        self.sign_unique_single_up = torch.as_tensor(
            self.sign_unique_single_up).to(self.device)
        self.sign_unique_single_down = torch.as_tensor(
            self.sign_unique_single_down).to(self.device)

    def get_index_unique_double(self):
        """Computes the 1D index of the double excitation matrices."""

        ncol_up = self.max_orb[0]-self.nup
        ncol_down = self.max_orb[1]-self.ndown

        self.index_unique_double_up = []
        self.index_unique_double_down = []

        for exc_up, exc_down in zip(self.unique_excitations[0],
                                    self.unique_excitations[1]):

            if len(exc_up[0]) == 2:
                for ielec in exc_up[0]:
                    for iorb in exc_up[1]:
                        icol = iorb-self.nup
                        self.index_unique_double_up.append(
                            ielec*ncol_up + icol)

            if len(exc_down[1]) == 2:
                for ielec in exc_up[0]:
                    for iorb in exc_up[1]:
                        icol = iorb-self.ndown
                        self.index_unique_double_down.append(
                            ielec*ncol_down + icol)
