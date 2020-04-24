import torch
from torch import nn

from .orbital_projector import OrbitalProjector


def btrace(M):
    return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)


def bproj(M, P):
    return P.transpose(1, 2) @ M @ P


class KineticPooling(nn.Module):

    def __init__(self, configs, mol, cuda=False):
        """Computes the kinetic energy using the Jacobi formula

        Args:
            configs (list): list of slater determinants
            mol (Molecule): instance of a Molecule object
            cuda (bool, optional): GPU ON/OFF. Defaults to False.
        """

        super(KineticPooling, self).__init__()

        self.configs = configs
        self.nconfs = len(configs[0])

        self.nup = mol.nup
        self.ndown = mol.ndown
        self.nelec = self.nup + self.ndown

        self.orb_proj = OrbitalProjector(configs, mol)

        if cuda:
            self.device = torch.device('cuda')
            self.orb_proj.Pup = self.orb_proj.Pup.to(self.device)
            self.orb_proj.Pdown = self.orb_proj.Pdown.to(self.device)

    def forward(self, MO, d2MO, dJdMO=None, d2JMO=None):
        """Compute the kinetic energy using the trace trick for a product of spin up/down determinant.

        .. math::
            -\\frac{1}{2} \Delta \Psi = -\\frac{1}{2}  D_{up} D_{down}  
            ( \Delta_{up} D_{up}  + \Delta_{down} D_{down} )

        Args:
            MO (torch.tensor): matrix of MO vals(Nbatch, Nelec, Nmo)
            d2MO (torch.tensor): matrix of :math:`\Delta` MO vals(Nbatch, Nelec, Nmo)
            dJdMO (torch.tensor, optional): matrix of the :math:`\\frac{\\nabla J}{J} \\nabla MO`. Defaults to None.
            d2JMO (torch.tensor, optional): matrix of the :math:`\\frac{\Delta J}{J} MO`. Defaults to None.

        Returns:
            torch.tensor: kinetic energy
        """

        # shortcut up/down matrices
        Aup, Adown = self.orb_proj.split_orbitals(MO)
        if dJdMO is None and d2JMO is None:
            Bup, Bdown = self.orb_proj.split_orbitals(d2MO)
        else:
            Bup, Bdown = self.orb_proj.split_orbitals(
                d2MO + 2 * dJdMO + d2JMO)

        # inverse of MO matrices
        iAup = torch.inverse(Aup)
        iAdown = torch.inverse(Adown)

        # determinant product
        det_prod = torch.det(Aup) * torch.det(Adown)

        # kinetic terms
        kinetic = -0.5 * (btrace(iAup@Bup) +
                          btrace(iAdown@Bdown)) * det_prod

        # reshape
        kinetic = kinetic.transpose(0, 1)
        det_prod = det_prod.transpose(0, 1)

        return kinetic, det_prod
