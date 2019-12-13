import torch
from torch import nn
from torch.autograd import Variable

from deepqmc.wavefunction.orbital_projector import OrbitalProjector


class BatchDeterminant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):

        # LUP decompose the matrices
        inp_lu, pivots = input.lu()
        perm, inpl, inpu = torch.lu_unpack(inp_lu, pivots)

        # get the number of permuations
        s = (pivots != torch.tensor(
            range(1, input.shape[1]+1)).int()).sum(1).float()

        # get the prod of the diag of U
        d = torch.diagonal(inpu, dim1=-2, dim2=-1).prod(1)

        # assemble
        det = ((-1)**s * d)
        ctx.save_for_backward(input, det)

        return det

    @staticmethod
    def backward(ctx, grad_output):
        '''using jaobi's formula
            d det(A) / d A_{ij} = adj^T(A)_{ij}
        using the adjunct formula
            d det(A) / d A_{ij} = ( (det(A) A^{-1})^T )_{ij}
        '''
        input, det = ctx.saved_tensors
        return (grad_output * det).view(-1, 1, 1) * torch.inverse(input).transpose(1, 2)


class SlaterPooling(nn.Module):

    """Applies a slater determinant pooling in the active space."""

    def __init__(self, configs, mol):
        super(SlaterPooling, self).__init__()

        self.configs = configs
        self.nconfs = len(configs[0])

        self.nmo = mol.norb
        self.nup = mol.nup
        self.ndown = mol.ndown

        self.index_up = torch.arange(self.nup)
        self.index_down = torch.arange(self.nup, self.nup+self.ndown)

        self.orb_proj = OrbitalProjector(configs, mol)

    def forward(self, input):
        ''' Compute the product of spin up/down determinants
        Args:
            input : MO values (Nbatch, Nelec, Nmo)
        Returnn:
            determiant (Nbatch, Ndet)
        '''

        mo_up, mo_down = self.orb_proj.split_orbitals(input)
        return (torch.det(mo_up) * torch.det(mo_down)).transpose(0, 1)
        # .view(-1, self.nconfs)

    def _forward_loop(self, input):
        ''' Compute the product of spin up/down determinants
        Args:
            input : MO values (Nbatch, Nelec, Nmo)
        Returnn:
            determiant (Nbatch, Ndet)

        DEPRECATED
        '''
        nbatch = input.shape[0]
        out = torch.zeros(nbatch, self.nconfs)

        for ic, (cup, cdown) in enumerate(zip(self.configs[0], self.configs[1])):

            mo_up = input.index_select(1, self.index_up).index_select(2, cup)
            mo_down = input.index_select(
                1, self.index_down).index_select(2, cdown)

            # a batch version of det is on its way (end July 2019)
            # https://github.com/pytorch/pytorch/issues/7500
            # we'll move to that asap but in the mean time
            # using my own BatchDeterminant
            try:
                out[:, ic] = torch.det(mo_up) * torch.det(mo_down)
            except:
                out[:, ic] = BatchDeterminant.apply(
                    mo_up) * BatchDeterminant.apply(mo_down)

        return out


if __name__ == "__main__":

    x = Variable(torch.rand(10, 5, 5))
    x.requires_grad = True
    det = BatchDeterminant.apply(x)
    det.backward(torch.ones(10))

    det_true = torch.tensor([torch.det(xi).item() for xi in x])
    print(det-det_true)
