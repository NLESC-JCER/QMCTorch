import torch
from torch import nn
from torch.autograd import grad, Variable


class OrbitalDependentBackFlowKernel(nn.Module):

    def __init__(self, backflow_kernel, mol, cuda):
        """Compute the back flow kernel, i.e. the function
        f(rij) where rij is the distance between electron i and j
        This kernel is used in the backflow transformation
        .. math:
            q_i = r_i + \\sum_{j\\neq i} f(r_{ij}) (r_i-r_j)
        """
        super().__init__()

        self.nelec = mol.nelec
        self.nao = mol.basis.nao
        self.orbital_dependent_kernel = nn.ModuleList(
            [backflow_kernel(mol, cuda) for iao in range(self.nao)])

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('gpu')

        self.stack_axis = 1

    def forward(self, ree, derivative=0):
        """Computes the desired values of the kernel
         Args:
            ree (torch.tensor): e-e distance Nbatch x Nelec x Nelec
            derivative (int): derivative requried 0, 1, 2

        Returns:
            torch.tensor : f(r) Nbatch x Nelec x Nelec
        """
        out = None

        for ker in self.orbital_dependent_kernel:
            ker_val = ker(ree, derivative).unsqueeze(self.stack_axis)
            if out is None:
                out = ker_val
            else:
                out = torch.cat((out, ker_val), axis=self.stack_axis)
        return out
