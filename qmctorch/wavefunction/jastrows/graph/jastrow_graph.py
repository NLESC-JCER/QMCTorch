import torch
from torch import nn
from ..distance.electron_electron_distance import ElectronElectronDistance
from ..distance.electron_nuclei_distance import ElectronNucleiDistance


class JastrowFactorGraph(nn.Module):

    def __init__(self, nup, ndown,
                 atomic_pos,
                 network,
                 network_kwargs={},
                 cuda=False):
        """Graph Neural Network Jastrow Factor

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            atomic_pos(torch.tensor): positions of the atoms
            network (dgl model): graph network of the factor
            network_kwargs (dict, optional): Argument of the graph network. Defaults to {}.
            cuda (bool, optional): use cuda. Defaults to False.
        """

        super().__init__()

        self.nup = nup
        self.ndown = ndown
        self.nelec = nup + ndown
        self.ndim = 3

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

        self.atoms = atomic_pos.to(self.device)
        self.natoms = atomic_pos.shape[0]

        self.requires_autograd = True

        # mask to extract the upper diag of the matrices
        self.mask_tri_up, self.index_col, self.index_row = self.get_mask_tri_up()

        # distance calculator
        self.elel_dist = ElectronElectronDistance(self.nelec,
                                                  self.ndim)
        self.elnu_dist = ElectronNucleiDistance(self.nelec,
                                                self.atoms, self.ndim)

        self.model = network(**network_kwargs)

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
                          derivative = 2  (Nmo) x Nbatch x Nelec
        """

        size = pos.shape
        assert size[1] == self.nelec * self.ndim
        nbatch = size[0]

    def get_mask_tri_up(self):
        r"""Get the mask to select the triangular up matrix

        Returns:
            torch.tensor: mask of the tri up matrix
        """
        mask = torch.zeros(self.nelec, self.nelec).type(
            torch.bool).to(self.device)
        index_col, index_row = [], []
        for i in range(self.nelec-1):
            for j in range(i+1, self.nelec):
                index_row.append(i)
                index_col.append(j)
                mask[i, j] = True

        index_col = torch.LongTensor(index_col).to(self.device)
        index_row = torch.LongTensor(index_row).to(self.device)
        return mask, index_col, index_row

    def extract_tri_up(self, inp):
        r"""extract the upper triangular elements

        Args:
            input (torch.tensor): input matrices (..., nelec, nelec)

        Returns:
            torch.tensor: triangular up element (..., nelec_pair)
        """
        shape = list(inp.shape)
        out = inp.masked_select(self.mask_tri_up)
        return out.view(*(shape[:-2] + [-1]))

    def extract_elec_nuc_dist(self, en_dist):
        r"""Organize the elec nuc distances

        Args:
            en_dist (torch.tensor): electron-nuclei distances
                                    nbatch x nelec x natom or
                                    nbatch x 3 x nelec x natom (dr)

        Returns:
            torch.tensor: nbatch x natom x nelec_pair x 2 or
            torch.tensor: nbatch x 3 x natom x nelec_pair x 2 (dr)
        """
        out = en_dist[..., self.index_elec, :]
        if en_dist.ndim == 3:
            return out.permute(0, 3, 2, 1)
        elif en_dist.ndim == 4:
            return out.permute(0, 1, 4, 3, 2)
        else:
            raise ValueError(
                'elec-nuc distance matrix should have 3 or 4 dim')

    def assemble_dist(self, pos):
        """Assemle the different distances for easy calculations

        Args:
            pos (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim

        Returns:
            torch.tensor : nbatch, natom, nelec_pair, 3

        """

        # get the elec-elec distance matrix
        ree = self.extract_tri_up(self.elel_dist(pos))
        ree = ree.unsqueeze(1).unsqueeze(-1)
        ree = ree.repeat(1, self.natoms, 1, 1)

        # get the elec-nuc distance matrix
        ren = self.extract_elec_nuc_dist(self.elnu_dist(pos))

        # cat both
        return torch.cat((ren, ree), -1)

    def assemble_dist_deriv(self, pos, derivative=1):
        """Assemle the different distances for easy calculations
           the output has dimension  nbatch, 3 x natom, nelec_pair, 3
           the last dimension is composed of [r_{e_1n}, r_{e_2n}, r_{ee}]

        Args:
            pos (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim

        Returns:
            torch.tensor : nbatch, 3 x natom, nelec_pair, 3

        """

        # get the elec-elec distance derivative
        dree = self.elel_dist(pos, derivative)
        dree = self.extract_tri_up(dree)
        dree = dree.unsqueeze(2).unsqueeze(-1)
        dree = dree.repeat(1, 1, self.natoms, 1, 1)

        # get the elec-nuc distance derivative
        dren = self.elnu_dist(pos, derivative)
        dren = self.extract_elec_nuc_dist(dren)

        # assemble
        return torch.cat((dren, dree), -1)
