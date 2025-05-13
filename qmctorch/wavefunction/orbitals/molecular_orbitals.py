import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal
from ...scf import Molecule


class MolecularOrbitals(nn.Module):
    def __init__(
        self,
        mol: Molecule,
        include_all_mo: bool,
        highest_occ_mo: int,
        mix_mo: bool,
        orthogonalize_mo: bool,
        cuda: bool,
    ):
        """
        Args:
            mol (Molecule): molecule object
            include_all_mo (bool): If True all molecular orbitals are included in the optimization
            highest_occ_mo (int): If include_all_mo=False, the highest occupied molecular orbital that is included in the optimization
            mix_mo (bool): If True, the molecular orbitals are mixed according to the mixing weights
            orthogonalize_mo (bool): If True, the mixing weights are orthogonalized
            cuda (bool): If True, the computation is done on the GPU, if not, it is done on the CPU
        """
        super(MolecularOrbitals, self).__init__()
        dtype = torch.get_default_dtype()

        self.mol = mol
        self.mix_mo = mix_mo
        self.orthogonalize_mo = orthogonalize_mo

        self.cuda = cuda
        self.device = torch.device("cpu")
        if self.cuda:
            self.device = torch.device("cuda")

        self.include_all_mo = include_all_mo
        self.highest_occ_mo = highest_occ_mo
        self.nmo_opt = self.mol.basis.nmo if include_all_mo else self.highest_occ_mo

        self.mo_scf = self.get_mo_coeffs()
        self.mo_modifier = nn.Parameter(
            torch.ones_like(self.mo_scf, requires_grad=True)
        ).type(dtype)

        self.mo_mixer = None
        if self.mix_mo:
            self.mo_mixer.weight = nn.Parameter(torch.eye(self.nmo_opt, self.nmo_opt))
            if self.orthogonalize_mo:
                self.mo_mixer = orthogonal(self.mo_mixer)

        if orthogonalize_mo and (not mix_mo):
            raise Warning("orthogonalize_mo=True has no effect as mix_mo=False")

        if orthogonalize_mo:
            raise Warning("Option orthogonalize_mo will be dprecated in 0.5.0")

        if self.cuda:
            self.mo_scf = self.mo_scf.to(self.device)
            self.mo_modifier.to(self.device)
            if self.mix_mo:
                self.mo_mixer.to(self.device)

    def get_mo_coeffs(self) -> torch.tensor:
        """
        Returns the molecular orbital coefficients.
        If include_all_mo=True, all the molecular orbitals are returned.
        If include_all_mo=False, only the highest occupied molecular orbitals are returned.

        Returns:
            torch.tensor: Molecular orbital coefficients (Nmo, Nao)
        """
        mo_coeff = torch.as_tensor(self.mol.basis.mos).type(torch.get_default_dtype())
        if not self.include_all_mo:
            mo_coeff = mo_coeff[:, : self.highest_occ_mo]
        return mo_coeff.requires_grad_(False)

    def forward(self, ao: torch.tensor) -> torch.tensor:
        """
        Transforms atomic orbital values into molecular orbital values using
        the molecular orbital coefficients, mo modifier and optinally a mixed.

        Args:
            ao (torch.tensor): Atomic orbital values (Nbatch, Nelec, Nao).

        Returns:
            torch.tensor: Transformed molecular orbital values (Nbatch, Nelec, Nmo).
        """

        weight = self.mo_scf * self.mo_modifier
        out = ao @ weight.reshape(1, *weight.shape)
        if self.mix_mo:
            out = self.mo_mixer(out)
        return out
