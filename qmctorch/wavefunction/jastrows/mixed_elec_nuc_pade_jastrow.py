import torch
from torch import nn

from .elec_elec.old_files.pade_jastrow import PadeJastrow
from .elec_nuclei.electron_nuclei_pade_jastrow import ElectronNucleiPadeJastrow


class MixedElecNucPadeJastrow(nn.Module):

    def __init__(self, nup, ndown, atomic_pos,  wee=1., wen=1., cuda=False):
        r"""Computes the Simple Pade-Jastrow factor

        .. math::
            J = \prod{\alpha,i} \exp(B_{ij}) \prod_{i<j} \exp(B_{ij}) \quad \quad \\text{with} \quad \quad
            B_{ij} = \\frac{w_0 r_{i,j}}{1 + w r_{i,j}}

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            atoms (torch.tensor): atomic positions of the atoms
            w (float, optional): Value of the variational parameter. Defaults to 1..
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """
        super(MixedElecNucPadeJastrow, self).__init__()
        self.elec_nuc = ElectronNucleiPadeJastrow(
            nup, ndown, atomic_pos, w=wen, cuda=cuda)
        self.elec_elec = PadeJastrow(nup, ndown, w=wee, cuda=cuda)

    def forward(self, pos, derivative=0, sum_grad=True):
        """Compute the Jastrow factors.

        Args:
            pos (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0,1,2,).
                            Defaults to 0.
            sum_grad (bool, optional): Return the sum_grad (i.e. the sum of
                                       the derivatives) or the individual
                                       terms. Defaults to True.
                                       False only for derivative=1

        Returns:
            torch.tensor: value of the jastrow parameter for all confs
                          derivative = 0  (Nmo) x Nbatch x 1
                          derivative = 1  (Nmo) x Nbatch x Nelec (for sum_grad = True)
                          derivative = 1  (Nmo) x Nbatch x Ndim x Nelec (for sum_grad = False)
        """

        if derivative == 0:
            return self.elec_elec(pos) * self.elec_nuc(pos)

        elif derivative == 1:

            ee_jast = self.elec_elec(pos)
            der_ee_jast = self.elec_elec(
                pos, derivative=1, sum_grad=sum_grad)

            en_jast = self.elec_nuc(pos)
            der_en_jast = self.elec_nuc(
                pos, derivative=1, sum_grad=sum_grad)

            if sum_grad:
                return der_ee_jast*en_jast + ee_jast*der_en_jast
            else:
                return der_ee_jast*en_jast.unsqueeze(-1) + ee_jast*der_en_jast.unsqueeze(-1)

        elif derivative == 2:
            ee_jast, der_ee_jast, der2_ee_jast = self.elec_elec(
                pos, derivative=[0, 1, 2], sum_grad=False)

            en_jast, der_en_jast, der2_en_jast = self.elec_nuc(
                pos, derivative=[0, 1, 2], sum_grad=False)

            return der2_ee_jast * en_jast + ee_jast * der2_en_jast + 2 * (der_ee_jast*der_en_jast).sum(1)
