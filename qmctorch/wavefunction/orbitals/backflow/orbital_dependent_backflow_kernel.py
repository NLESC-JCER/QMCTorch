import torch
from torch import nn


class OrbitalDependentBackFlowKernel(nn.Module):

    def __init__(self, backflow_kernel, backflow_kernel_kwargs, mol, cuda):
        """Compute orbital dependent back flow kernel, i.e. the functions
        f(rij) where rij is the distance between electron i and j
        This kernel is used in the backflow transformation
        .. math:
            q^{\\alpha}_i = r_i + \\sum_{j\\neq i} f^{\\alpha}(r_{ij}) (r_i-r_j)

        where :math: `f^{\\alpha}(r_{ij})` is the kernel for obital :math: `\\alpha`
        """
        super().__init__()

        self.nelec = mol.nelec
        self.nao = mol.basis.nao
        self.orbital_dependent_kernel = nn.ModuleList(
            [backflow_kernel(mol, cuda, **backflow_kernel_kwargs) for iao in range(self.nao)])

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

        # domension along which the different orbitals are stacked
        # with stach_axis = 1 the resulting tensors will have dimension
        # Nbatch x Nao x ...
        self.stack_axis = 1

    def forward(self, ree, derivative=0):
        """Computes the desired values of the kernels
         Args:
            ree (torch.tensor): e-e distance Nbatch x Nelec x Nelec
            derivative (int): derivative requried 0, 1, 2

        Returns:
            torch.tensor : f(r) Nbatch x Nao x Nelec x Nelec
        """
        out = None

        for ker in self.orbital_dependent_kernel:
            ker_val = ker(ree, derivative).unsqueeze(self.stack_axis)
            if out is None:
                out = ker_val
            else:
                out = torch.cat((out, ker_val), axis=self.stack_axis)
        return out
