

class FermiPrepocess(nn.Module):

    def __init__(self, mol):
        """preprocessing of the electron position to the input of the first layer.

        Args:
            mol (Molcule instance): Molecule object
        """
        self.mol = mol

    def forward(self, pos):

        # input of electron and nuclei positions
        r_electrons = pos.clone().detach().requires_grad_(True)
        R_Nuclei = torch.tensor(self.mol.atom_coords)

        # Look at the electron nuclei distances
        h_i = self.electron_nuclei_input(r_electrons, self.mol)

        # Now look at the electron electron distances
        h_ij = self.electron_electron_input(r_electrons, self.mol)

        return h_i, h_ij

    @staticmethod
    def electron_nuclei_input(pos, mol):
        '''Function to create intial input of electron-nuclei distances.

        Args:
            pos : electron positions Nbatch x [Nelec x Ndim]
            mol (object) : Molecule instance

        '''
        # input of electron and molecule object for the atom positions
        h_0_i = torch.tensor([])
        # measure the distances between electrons and the nuclei,
        # and determine the input for the single electron stream
        R_nuclei = torch.tensor(mol.atom_coords)
        for l in range(mol.natom):
            r_il = (pos-R_nuclei[l, None]).clone()
            r_il_len = torch.norm(r_il, dim=2).reshape(
                pos.shape[0], pos.shape[1], 1)
            h_0_il = torch.cat((r_il, r_il_len), axis=2)
            h_0_i = torch.cat(
                (h_0_i, h_0_il), axis=2) if h_0_i.size else h_0_il
        return h_0_i

    @staticmethod
    def electron_electron_input(pos, mol):
        '''Function to create intial input of electron-electron distances.

        Args:
            pos : electron positions Nbatch x [Nelec x Ndim]
            mol (object) : Molecule instance

        '''
        nbatch = pos.shape[0]
        # create meshgrid for indexing of electron i and j
        [i, j] = torch.meshgrid(torch.arange(
            0, mol.nelec), torch.arange(0, mol.nelec))
        xij = torch.zeros((nbatch, mol.nelec, mol.nelec, 3))

        # determine electron - electron distance vector
        xij[:, :, :, :] = (pos[:, i, :]-pos[:, j, :]).reshape(
            nbatch, mol.nelec, mol.nelec, 3)

        # determine absolute distance
        rij = torch.norm(xij, dim=3)
        h_0_ij = torch.cat(
            (xij, rij.reshape(nbatch, mol.nelec, mol.nelec, 1)), axis=3)
        # output h_0_ij Nbatch x Nelec x Nelec x Ndim+1
        return h_0_ij


class FermiIntermediate(nn.Module):

    def __init__(self, mol, input_size_e, output_size_e, input_size_ee, output_size_ee):
        """Implementatation of a single intermediate layer."""

        self.mol = mol
        self.lin_layer_e = nn.Linear(
            input_size, output_size, bias=True)
        self.lin_layer_ee = nn.Linear(
            input_size_ee, output_size_ee, bias=True)

    def forward(self, h_i, h_ij):

        # for the one electron stream:
        h_i_previous = h_i
        f = self.f_concatenate(h_i, h_ij, self.mol)
        l_e = self.lin_layer_e(f)

        # with a tanh activation and dependent on the hidden layers size a residual connection
        h_i = torch.tanh(l_e)
        if h_i.shape[2] == h_i_previous.shape[2]:
            h_i = h_i + h_i_previous
        # for the two electron stream:
        h_ij_previous = h_ij
        l_ee = self.lin_layer_ee(h_ij)
        # with a tanh activation and dependent on the hidden layers size a residual connection
        h_ij = torch.tanh(l_ee)
        if h_ij.shape[3] == h_ij_previous.shape[3]:
            h_ij = h_ij + h_ij_previous

        return h_i, h_ij

    @staticmethod
    def f_concatenate(h_i, h_ij, mol):
        '''Function to concatenate the desired information to get the input for each new layer.
        With as input the one electron and two electron stream output of the previous layer and
        the spin assignment of the electrons.

        Args: 
            h_i : one electron stream output of previous layer Nbatch x Nelec x hidden_nodes_e
            h_ij: two electron stream output of previous layer Nbatch x Nelec x Nelec x hidden_nodes_ee
            mol : Molecule instance 
        '''
        nbatch = h_i.shape[0]
        hidden_nodes_e = h_i.shape[2]

        g_down = torch.mean(h_i[:, :mol.nup], axis=1).repeat(
            mol.nelec, 1).reshape(nbatch, mol.nelec, hidden_nodes_e)
        g_up = torch.mean(h_i[:, mol.nup:], axis=1).repeat(
            mol.nelec, 1).reshape(nbatch, mol.nelec, hidden_nodes_e)
        g_down_i = torch.mean(h_ij[:, :mol.nup], axis=1)
        g_up_i = torch.mean(h_ij[:, mol.nup:], axis=1)

        f_i = torch.cat((h_i, g_down, g_up, g_down_i, g_up_i), axis=2)
        # outputs a array f_i (N_batch x Nelec x f_size)
        return f_i

    @staticmethod
    def f_size(hidden_nodes_e, hidden_nodes_ee):
        '''Function to get the input size for the hidden layers'''
        return 3*hidden_nodes_e + 2*(hidden_nodes_ee)


class FermiOrbital(nn.Module):

    def __init__(self, mol, Kdet, input_size):
        """Computes all the orbitals from the output of the last layer."""

        self.mol = mol
        self.Kdet = Kdet

        self.W = nn.Parameters(torch.rand(
            self.Kdet, input_size, self.mol.norb))

        self.G = nn.Parameters(torch.rand(
            Kdet, self.mol.nelec, self.mol.norb))

    def forward(self, h_i, r_i):

        out = h_i @ self.W + self.G
        out = out * (exponential stuff)

        return out


class FermiNet(nn.Module):

    def __init__(self, mol, Kdet):

        self.prepoc = FermiPrepocess(mol)

        inp_e1, out_e1 = ...
        inp_ee1, out_ee1 = ...
        self.layer1 = FermiIntermediate(
            mol, inp_e1, out_e1, inp_ee_1, out_ee1)

        inp_e2, out_e2 = ...
        inp_ee2, out_ee2 = ...
        self.layer1 = FermiIntermediate(
            mol, inp_e2, out_e2, inp_ee_2, out_ee2)

        self.orb = FermiOrbital(mol, Kdet, out_e2)

    def forward(self, pos):

        h_i, h_ij = self.prepoc(pos)
        h_i, h_ij = self.layer1(h_i, h_ij)
        h_i, h_ij = self.layer2(h_i, h_ij)
        orbs = self.orb(h_i, pos)
        return torch.det(orb)
