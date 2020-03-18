import torch
from torch import nn

from deepqmc.wavefunction.orbital_projector import OrbitalProjector


def btrace(M):
    return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)


def bproj(M, P):
    return P.transpose(1, 2) @ M @ P


class KineticPooling(nn.Module):

    def __init__(self, configs, mol, cuda=False):
        """Layer that computes the kinetic energy using the jacobi formula (trace trick)

        Arguments:
            configs {list} -- configuration of the slater determinant
            mol {[type]} -- Instance of a Molecule object

        Keyword Arguments:
            cuda {bool} -- use cuda (default: {False})
        """
        super(KineticPooling, self).__init__()

        self.configs = configs
        self.nconfs = len(configs[0])

        self.nup = mol.nup
        self.ndown = mol.ndown
        self.nelec = self.nup+self.ndown

        self.orb_proj = OrbitalProjector(configs, mol)

        if cuda:
            self.device = torch.device('cuda')
            self.orb_proj.Pup = self.orb_proj.Pup.to(self.device)
            self.orb_proj.Pdown = self.orb_proj.Pdown.to(self.device)

    def forward(self, MO, d2MO, dJdMO=None, d2JMO=None):
        """ Compute the kinetic energy using the trace trick
        for a product of spin up/down determinant
        .. math::

            T \Psi  =  T Dup Ddwn
                    = -1/2 Dup * Ddown  * ( \Delta_up Dup  + \Delta_down Ddown)

            using the trace trick with D = |A| :
                O(D) = D trace(A^{-1} O(A))
                and Delta_up(D_down) = 0

        Args:
            MO : matrix of MO vals (Nbatch, Nelec, Nmo)
            d2MO : matrix of \Delta MO vals (Nbatch, Nelec, Nmo)
            dJdMO : matrix of the \frac{\nabla J}{J} \nabla MO
            d2JMO : matrix of the \frac{\Delta J}{J} MO
            return_local_energy : divide the contrbutions by det(MO) to get
                                  local energy instead of kinetic energy
        Return:
            K : T Psi (Nbatch, Ndet)
        """

        # shortcut up/down matrices
        Aup, Adown = self.orb_proj.split_orbitals(MO)
        if dJdMO is None and d2JMO is None:
            Bup, Bdown = self.orb_proj.split_orbitals(d2MO)
        else:
            Bup, Bdown = self.orb_proj.split_orbitals(
                d2MO + 2*dJdMO + d2JMO)

        # inverse of MO matrices
        iAup = torch.inverse(Aup)
        iAdown = torch.inverse(Adown)

        # determinant product
        det_prod = torch.det(Aup) * torch.det(Adown)

        # kinetic terms
        kinetic = -0.5*(btrace(iAup@Bup) + btrace(iAdown@Bdown)) * det_prod

        # reshape
        kinetic = kinetic.transpose(0, 1)
        det_prod = det_prod.transpose(0, 1)

        return kinetic, det_prod
