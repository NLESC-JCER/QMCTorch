#In this file the intermediate layers of the Fermionic Neural Network preposed in [arXiv:1909.02487].
import torch 
import torch.nn as nn

class IntermediateLayers(nn.Module):

    """ Creates the intermediate layers of the Fermi net with required concatenations."""
    
    def __init__(self, mol, nspin, hidden_nodes_e, hidden_nodes_ee, L_layers, ndim=3 ,cuda=False):
        """Computes the intermediate layers of the Fermi Net.
         
        Args:
            mol (Molecule): Molecule instance
            nspin (int array): Spin orientation given as [n_up,n_down]
            ndim (int, optional): Number of dimensions.
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
            hidden_nodes_e (int): Number of hidden nodes for one electron stream
            hidden_nodes_ee (int): Number of hidden nodes for two electron stream
            L_layers (int): Number of hidden layers
        """
        super(IntermediateLayers, self).__init__()


        # molecule instance contains the required molecule information such as:
        # number of electrons
        # number of atoms
        # atomic nuclei coordinates

        # number of atoms and electrons
        self.mol = mol 
        self.atoms = mol.atoms
        self.natom = mol.natom
        self.nelec = mol.nelec

        # spin orientation of electrons 
        self.nspin = nspin

        # number of hidden layers 
        self.L_layers = L_layers

        # initial input size based on the concatenation
        input_size = (3*self.natom+2)*(ndim+1)

        # linear input layer
        self.lin_layer_e = nn.ModuleList()
        self.lin_layer_e.append(
            nn.Linear(input_size, hidden_nodes_e, bias=True))
        for l in range(1, self.L_layers):
            self.lin_layer_e.append(
                nn.Linear(self.f_size(hidden_nodes_e, hidden_nodes_ee), hidden_nodes_e, bias=True))

        self.lin_layer_ee = nn.ModuleList()
        self.lin_layer_ee.append(
            nn.Linear(ndim+1, hidden_nodes_ee, bias=True))
        for l in range(1, self.L_layers):
            self.lin_layer_ee.append(
                nn.Linear(hidden_nodes_ee, hidden_nodes_ee, bias=True))


    def forward(self, pos):
        """
            pos (float): electron coordinates
        """ 
        # input of electron and nuclei positions
        r_electrons = pos
        R_Nuclei = torch.tensor(self.mol.atom_coords).clone().detach()

        # Look at the electron nuclei distances
        h_i = self.electron_nuclei_input(r_electrons, R_Nuclei)

        # Now look at the electron electron distances
        h_ij = self.electron_electron_input(r_electrons)

        # now that we have the desired interaction distances we will concatenate to f

        for l in range(self.L_layers):
            # for the one electron stream:
            h_i_previous = h_i
            f = self.f_concatenate(h_i, h_ij, self.nspin)
            l_e = self.lin_layer_e[l](f)
            # with a tanh activation and dependent on the hidden layers size a residual connection
            h_i = torch.tanh(l_e)
            if h_i.shape[1] == h_i_previous.shape[1]:
                h_i = h_i + h_i_previous
            # for the two electron stream:
            h_ij_previous = h_ij
            l_ee = self.lin_layer_ee[l](h_ij)
            # with a tanh activation and dependent on the hidden layers size a residual connection
            h_ij = torch.tanh(l_ee)
            if h_ij.shape[2] == h_ij_previous.shape[2]:
                h_ij = h_ij + h_ij_previous

        return h_i, h_ij


    @staticmethod
    def f_concatenate(h_i, h_ij, N_spin):
        '''Function to concatenate the desired information to get the input for each new layer.
        With as input the one electron and two electron stream output of the previous layer and
        the spin assignment of the electrons.'''
        g_down = torch.mean(h_i[:N_spin[0]], axis=0).repeat(h_i.shape[0], 1)
        g_up = torch.mean(h_i[N_spin[0]:], axis=0).repeat(h_i.shape[0], 1)
        g_down_i = torch.mean(h_ij[:N_spin[0]], axis=0)
        g_up_i = torch.mean(h_ij[N_spin[0]:], axis=0)
        f_i = torch.cat((h_i, g_down, g_up, g_down_i, g_up_i), axis=1)
        # outputs a array f_i where the first dimension are the electrons i
        return f_i
    
    @staticmethod
    def f_size(hidden_nodes_e, hidden_nodes_ee):
        '''Function to get the input size for the hidden layers'''
        return 3*hidden_nodes_e + 2*(hidden_nodes_ee)

    @staticmethod
    def electron_nuclei_input(r_electrons, R_Nuclei):
        '''Function to create intial input of electron-nuclei distances.'''
        # input of electron and nuclei positions
        h_0_i = torch.tensor([])
        # measure the distances between electrons and the nuclei,
        # and determine the input for the single electron stream
        for l in range(R_Nuclei.shape[0]):
            r_il = (r_electrons-R_Nuclei[l])
            r_il_len = torch.norm(r_il, dim=1).reshape(r_electrons.shape[0], 1)
            h_0_il = torch.cat((r_il, r_il_len), axis=1)
            h_0_i = torch.cat(
                (h_0_i, h_0_il), axis=1) if h_0_i.size else h_0_il
        return h_0_i
    
    @staticmethod
    def electron_electron_input(r_electrons):
        '''Function to create intial input of electron-electron distances.'''
        # input of electron positions

        # create meshgrid for indexing of electron i and j
        [i, j] = torch.meshgrid(torch.arange(
            0, r_electrons.shape[0]), torch.arange(0, r_electrons.shape[0]))
        xij = torch.zeros((r_electrons.shape[0], r_electrons.shape[0], 3))
        
        # determine electron - electron distance vector
        xij[:, :, :] = (
            r_electrons[i, :]-r_electrons[j, :]).reshape(r_electrons.shape[0], r_electrons.shape[0], 3)
        
        # determine absolute distance
        rij = torch.norm(xij, dim=2)
        h_0_ij = torch.cat(
            (xij, rij.reshape(r_electrons.shape[0], r_electrons.shape[0], 1)), axis=2)

        return h_0_ij



