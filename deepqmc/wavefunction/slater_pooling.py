
import torch
from torch import nn
from torch.autograd import Variable

from deepqmc.wavefunction.orbital_projector import OrbitalProjector


class SlaterPooling(nn.Module):

    """Applies a slater determinant pooling in the active space."""

    def __init__(self, configs, mol, cuda=False):
        """Layer that computes all the slater determinant from the MO matrix.

        Arguments:
            configs {list} -- slater configuration
            mol {Molecule} -- Instance of the Molecule object

        Keyword Arguments:
            cuda {bool} -- use cuda (default: {False})
        """
        super(SlaterPooling, self).__init__()

        self.configs = configs
        self.nconfs = len(configs[0])

        self.nmo = mol.basis.nmo
        self.nup = mol.nup
        self.ndown = mol.ndown

        self.orb_proj = OrbitalProjector(configs, mol)

        if cuda:
            self.device = torch.device('cuda')
            self.orb_proj.Pup = self.orb_proj.Pup.to(self.device)
            self.orb_proj.Pdown = self.orb_proj.Pdown.to(self.device)

    def forward(self, input, return_matrix=False):
        """Computes the SD values

        Arguments:
            input {torch.tensor} -- MO matrices nbatc x nelec x nmo

        Keyword Arguments:
            return_matrix {bool} -- if true return the slater matrices (default: {False})

        Returns:
            torch.tensor -- slater matrices or determinant depending on return_matrix
        """

        mo_up, mo_down = self.orb_proj.split_orbitals(input)
        if return_matrix:
            return mo_up, mo_down
        else:
            return (torch.det(mo_up) * torch.det(mo_down)).transpose(0, 1)


if __name__ == "__main__":

    x = Variable(torch.rand(10, 5, 5))
    x.requires_grad = True
    det = BatchDeterminant.apply(x)
    det.backward(torch.ones(10))

    det_true = torch.tensor([torch.det(xi).item() for xi in x])
    print(det-det_true)
