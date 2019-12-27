import torch
from torch import nn

from deepqmc.wavefunction.orbital_projector import OrbitalProjector


def btrace(M):
    return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)


def bproj(M, P):
    return P.transpose(1, 2) @ M @ P


class KineticPooling(nn.Module):

    """Computes the kinetic energy of each configuration
       using the trace trick."""

    def __init__(self, configs, mol, cuda=False):
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

    def forward(self, MO, d2MO, dJdMO=None, d2JMO=None,
                return_local_energy=False):
        ''' Compute the kinetic energy using the trace trick
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
        '''

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

        # product
        out = (btrace(iAup@Bup) + btrace(iAdown@Bdown))

        # multiply by det if necessary
        if not return_local_energy:
            out *= torch.det(Aup) * torch.det(Adown)

        return -0.5*out.transpose(0, 1)

    def _forward_loop(self, MO, d2MO, dJdMO=None, d2JMO=None,
                      return_local_energy=False):
        ''' Compute the kinetic energy using the trace trick
        for a product of spin up/down determinant
        DEPRECATED  keep just in case
        .. math::

            T \Psi  =  T Dup Ddwn
                    = -1/2 Dup * Ddown  * ( \Delta_up Dup  + \Delta_down Ddown)

            using the trace trick with D = |A| :
                O(D) = D trace(A^{-1} O(A))
                and Delta_up(D_down) = 0

        Args:
            A : matrix of MO vals (Nbatch, Nelec, Nmo)
            d2A : matrix of \Delta MO vals (Nbatch, Nelec, Nmo)
        Return:
            K : T Psi (Nbatch, Ndet)
        '''

        nbatch = MO.shape[0]
        out = torch.zeros(nbatch, self.nconfs)

        for ic, (cup, cdown) in enumerate(zip(self.configs[0], self.configs[1])):

            Aup = MO.index_select(1, self.index_up).index_select(2, cup)
            Adown = MO.index_select(1, self.index_down).index_select(2, cdown)

            iAup = torch.inverse(Aup)
            iAdown = torch.inverse(Adown)

            d2Aup = d2MO.index_select(1, self.index_up).index_select(2, cup)
            d2Adown = d2MO.index_select(
                1, self.index_down).index_select(2, cdown)
            out[:, ic] = (btrace(iAup@d2Aup) + btrace(iAdown@d2Adown))

            if not return_local_energy:
                pd = torch.det(Aup) * torch.det(Adown)
                out[:, ic] *= pd

        return -0.5*out
