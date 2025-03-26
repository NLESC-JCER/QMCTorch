import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal 
from ...scf import Molecule
class MolecularOrbitals(nn.Module):
    def __init__(self, 
                 mol: Molecule, 
                 include_all_mo: bool, 
                 highest_occ_mo: int, 
                 mix_mo: bool, 
                 orthogonalize_mo: bool, 
                 cuda: bool):
        
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
        self.mo_modifier = nn.Parameter(torch.ones_like(self.mo_scf, requires_grad=True)).type(dtype)

        self.mo_mixed = None
        if self.mix_mo:
            self.mo_mixer.weight = nn.Parameter(torch.eye(self.nmo_opt, self.nmo_opt))
            if self.orthogonalize_mo:
                self.mo_mixer = orthogonal(self.mo_mixer)

        if self.cuda:
            self.mo_scf = self.mo_scf.to(self.device)
            self.mo_modifier.to(self.device)
            if self.mix_mo:
                self.mo_mixer.to(self.device)

    def get_mo_coeffs(self) -> torch.tensor:
        """Get the molecular orbital coefficients to init the mo layer."""
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
        out = ao @ weight.reshape(1,*weight.shape)
        if self.mix_mo:
            out = self.mo_mixer(out)
        return out