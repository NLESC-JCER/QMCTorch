import torch
from typing import List, Tuple
from ...scf import Molecule


class OrbitalProjector:
    def __init__(
        self, configs: List[torch.tensor], mol: Molecule, cuda: bool = False
    ) -> None:
        """Project the MO matrix in Slater Matrices

        Args:
            configs (List[torch.tensor]): configurations of the slater determinants
            mol (Molecule): Molecule object
            cuda (bool): use cuda or not
        """

        self.configs = configs
        self.nconfs = len(configs[0])
        self.nmo = mol.basis.nmo
        self.nup = mol.nup
        self.ndown = mol.ndown

        self.device = torch.device("cpu")
        if cuda:
            self.device = torch.device("cuda")
        self.unique_configs, self.index_unique_configs = self.get_unique_configs()

    def get_unique_configs(
        self,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Get the unique configurations

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
                configs_up (torch.Tensor): unique configurations of the spin up electrons
                configs_down (torch.Tensor): unique configurations of the spin down electrons
                index_unique_confs_up (torch.Tensor): index of the unique configurations of the spin up electrons
                index_unique_confs_down (torch.Tensor): index of the unique configurations of the spin down electrons
        """
        configs_up, index_unique_confs_up = torch.unique(
            self.configs[0], dim=0, return_inverse=True
        )
        configs_down, index_unique_confs_down = torch.unique(
            self.configs[1], dim=0, return_inverse=True
        )

        return (configs_up.to(self.device), configs_down.to(self.device)), (
            index_unique_confs_up.to(self.device),
            index_unique_confs_down.to(self.device),
        )

    def split_orbitals(
        self, mat: torch.Tensor, unique_configs: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split the orbital matrix in multiple Slater matrices
           This version does not store the projectors

        Args:
            mat: matrix to split
            unique_confgs: compute only the Slater matrices of the unique conf if True (Default=False)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: all Slater matrices
        """
        if mat.ndim == 3:
            nbatch = mat.shape[0]
            out_up = torch.zeros(0, nbatch, self.nup, self.nup, device=self.device)
            out_down = torch.zeros(
                0, nbatch, self.ndown, self.ndown, device=self.device
            )

        if mat.ndim == 4:
            nbatch = mat.shape[1]
            nop = mat.shape[0]
            out_up = torch.zeros(0, nop, nbatch, self.nup, self.nup, device=self.device)
            out_down = torch.zeros(
                0, nop, nbatch, self.ndown, self.ndown, device=self.device
            )

        if unique_configs:
            configs_up, configs_down = self.unique_configs

        else:
            configs_up, configs_down = self.configs

        for _, (cup, cdown) in enumerate(zip(configs_up, configs_down)):
            # cat the tensors
            out_up = torch.cat((out_up, mat[..., : self.nup, cup].unsqueeze(0)), dim=0)
            out_down = torch.cat(
                (out_down, mat[..., self.nup :, cdown].unsqueeze(0)), dim=0
            )

        return out_up, out_down


class ExcitationMask:
    def __init__(
        self,
        unique_excitations: List[Tuple[torch.Tensor, torch.Tensor]],
        mol: Molecule,
        max_orb: List[int],
        cuda: bool = False,
    ) -> None:
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

        self.device = torch.device("cpu")
        if cuda:
            self.device = torch.device("cuda")

    def get_index_unique_single(self) -> None:
        """Computes the 1D index and permutation
        for the unique singles."""

        ncol_up = self.max_orb[0] - self.nup
        ncol_down = self.max_orb[1] - self.ndown

        self.index_unique_single_up = []
        self.index_unique_single_down = []

        self.sign_unique_single_up = []
        self.sign_unique_single_down = []

        for exc_up, exc_down in zip(
            self.unique_excitations[0], self.unique_excitations[1]
        ):
            if len(exc_up[0]) == 1:
                ielec, iorb = exc_up[0][0], exc_up[1][0]
                icol = iorb - self.nup

                self.index_unique_single_up.append(ielec * ncol_up + icol)

                npermut = self.nup - ielec - 1
                self.sign_unique_single_up.append((-1) ** (npermut))

            if len(exc_down[1]) == 1:
                ielec, iorb = exc_down[0][0], exc_down[1][0]
                icol = iorb - self.ndown

                self.index_unique_single_down.append(ielec * ncol_down + icol)

                npermut = self.ndown - ielec - 1
                self.sign_unique_single_down.append((-1) ** (npermut))

        self.sign_unique_single_up = torch.as_tensor(self.sign_unique_single_up).to(
            self.device
        )
        self.sign_unique_single_down = torch.as_tensor(self.sign_unique_single_down).to(
            self.device
        )

    def get_index_unique_double(self) -> None:
        """Computes the 1D index of the double excitation matrices."""

        ncol_up = self.max_orb[0] - self.nup
        ncol_down = self.max_orb[1] - self.ndown

        self.index_unique_double_up = []
        self.index_unique_double_down = []

        for exc_up, exc_down in zip(
            self.unique_excitations[0], self.unique_excitations[1]
        ):
            if len(exc_up[0]) == 2:
                for ielec in exc_up[0]:
                    for iorb in exc_up[1]:
                        icol = iorb - self.nup
                        self.index_unique_double_up.append(ielec * ncol_up + icol)

            if len(exc_down[1]) == 2:
                for ielec in exc_up[0]:
                    for iorb in exc_up[1]:
                        icol = iorb - self.ndown
                        self.index_unique_double_down.append(ielec * ncol_down + icol)
