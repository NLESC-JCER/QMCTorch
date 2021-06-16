import torch
from torch import nn
from .jastrow_kernel_electron_electron_nuclei_base import JastrowKernelElectronElectronNucleiBase


class BoysHandyJastrowKernel(JastrowKernelElectronElectronNucleiBase):

    def __init__(self, nup, ndown, atomic_pos, cuda, nterm=5):
        """Defines a Boys Handy jastrow factors.

        J.W. Moskowitz et. al
        Correlated Monte Carlo Wave Functions for Some Cations and Anions of the First Row Atoms
        Journal of Chemical Physics, 97, 3382-85 (1992)

        .. math::
            \\text{K}(R_{iA}, R_{jA}, r_{ij) = \\sum_\\mu c_\\mu \\left(\\frac{a_{1_\\mu} R_{iA}}{1 + b_{1_\\mu}R_{iA}}\\right)^{u_\\mu}
                                                                 \\left(\\frac{a_{2_\\mu} R_{jA}}{1 + b_{2_\\mu}R_{iA}}\\right)^{v_\\mu}
                                                                 \\left(\\frac{a_{3_\\mu} r_{ij}}{1 + b_{3_\\mu}r_{ij}}\\right)^{w_\\mu}

        We restrict the parameters of the two electron-nucleus distance to be equal
        otherwise the jastrow factor is not permutation invariant

        """

        super().__init__(nup, ndown, atomic_pos, cuda)

        self.nterm = nterm
        self.fc = nn.Linear(self.nterm, 1, bias=False)

        self.weight_num = nn.Parameter(torch.rand(1, 2, self.nterm))
        self.weight_denom = nn.Parameter(torch.rand(1, 2, self.nterm))
        self.exp = nn.Parameter(torch.ones(2, self.nterm))
        self.repeat_dim = torch.as_tensor([2, 1]).to(self.device)

    def forward(self, x):
        """Compute the values of the kernel

        Args:
            x (torch.tensor): e-e and e-n distances distance (Nbatch, Natom, Nelec_pairs, 3)
                              the last dimension holds the values [R_{iA}, R_{jA}, r_{ij}]
                              where i,j are electron index in the pair and A the atom index.

        Returns:
            torch.tensor: values of the kernel (Nbatch, Natom, Nelec_pairs, 1)

        """

        # return (1E-5*x**3).prod(-1, keepdim=True)

        # reshape the input so that all elements
        # are considered independently of each other
        out_shape = list(x.shape)[:-1] + [1]

        # reshape to [N, 3 ,1]
        # here N is the
        x = x.reshape(-1, 3, 1)

        # compute the different terms :
        # x[0] = (a r_{iA})/(1 + b r_{iA})
        # x[1] = (a r_{jA})/(1 + b r_{jA})
        # x[2] = (a r_{ij})/(1 + b r_{ij})
        # output shape : [N, 3, nterm]
        wnum = self.weight_num.repeat_interleave(
            self.repeat_dim, dim=1)
        wdenom = self.weight_denom.repeat_interleave(
            self.repeat_dim, dim=1)
        x = (wnum * x) / (1. + wdenom * x)

        # comput the powers
        xp = self.exp.repeat_interleave(self.repeat_dim, dim=0)
        x = x**(xp)

        # product over the r_{iA}, r_{jA}, r_{ij}
        # output shape : [N, nterm]
        x = x.prod(1)

        # compute the sum over the different terms
        # output shape :  [N]
        x = self.fc(x)

        # print(x.reshape(*out_shape))
        return x.reshape(*out_shape)
