#In this file the intermediate layers of the Fermionic Neural Network preposed in [arXiv:1909.02487].
import torch 
import torch.nn as nn

class IntermediateLayers(nn.Module):

    """ Creates the intermediate layers of the Fermi net with required concatenations."""
    
    def __init__(self, mol, hidden_nodes_e, hidden_nodes_ee, L_layers, ndim=3 ,cuda=False):
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
        # number of spin up and spin down electrons
        # number of atoms
        # atomic nuclei coordinates

        # number of atoms and electrons
        self.mol = mol 
        self.atoms = mol.atoms
        self.natom = mol.natom
        self.nelec = mol.nelec

        # spin orientation of electrons 
        self.nup = mol.nup
        self.ndown = mol.ndown

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
            pos (float): electron coordinates  batch x nelectron x ndim
        """ 
        # input of electron and nuclei positions
        r_electrons = pos.clone().detach().requires_grad_(True)
        R_Nuclei = torch.tensor(self.mol.atom_coords)

        # Look at the electron nuclei distances
        h_i = self.electron_nuclei_input(r_electrons, self.mol)

        # Now look at the electron electron distances
        h_ij = self.electron_electron_input(r_electrons, self.mol)

        # now that we have the desired interaction distances we will concatenate to f

        for l in range(self.L_layers):
            # for the one electron stream:
            h_i_previous = h_i
            f = self.f_concatenate(h_i, h_ij, self.nup)
            l_e = self.lin_layer_e[l](f)
            # with a tanh activation and dependent on the hidden layers size a residual connection
            h_i = torch.tanh(l_e)
            if h_i.shape[2] == h_i_previous.shape[2]:
                h_i = h_i + h_i_previous
            # for the two electron stream:
            h_ij_previous = h_ij
            l_ee = self.lin_layer_ee[l](h_ij)
            # with a tanh activation and dependent on the hidden layers size a residual connection
            h_ij = torch.tanh(l_ee)
            if h_ij.shape[3] == h_ij_previous.shape[3]:
                h_ij = h_ij + h_ij_previous

        return h_i, h_ij


    @staticmethod
    def f_concatenate(h_i, h_ij, nup):
        '''Function to concatenate the desired information to get the input for each new layer.
        With as input the one electron and two electron stream output of the previous layer and
        the spin assignment of the electrons.
        
        Args: 
            h_i : one electron stream output of previous layer Nbatch x Nelec x hidden_nodes_e
            h_ij: two electron stream output of previous layer Nbatch x Nelec x Nelec x hidden_nodes_ee
            nup (int) : number of spin up electrons 
        '''
        Nbatch = h_i.shape[0]
        Nelec = h_i.shape[1]
        hidden_nodes_e = h_i.shape[2]

        g_down = torch.mean(h_i[:,:nup], axis=1).repeat(Nelec, 1).reshape(Nbatch, Nelec , hidden_nodes_e)
        g_up = torch.mean(h_i[:,nup:], axis=1).repeat(Nelec, 1).reshape(Nbatch, Nelec, hidden_nodes_e)
        g_down_i = torch.mean(h_ij[:,:nup], axis=1)
        g_up_i = torch.mean(h_ij[:,nup:], axis=1)

        f_i = torch.cat((h_i, g_down, g_up, g_down_i, g_up_i), axis=2)
        # outputs a array f_i (N_batch x Nelec x f_size)
        return f_i
    
    @staticmethod
    def f_size(hidden_nodes_e, hidden_nodes_ee):
        '''Function to get the input size for the hidden layers'''
        return 3*hidden_nodes_e + 2*(hidden_nodes_ee)

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
            r_il = (pos-R_nuclei[l,None]).clone()
            r_il_len = torch.norm(r_il, dim=2).reshape(pos.shape[0], pos.shape[1] , 1)
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
        Nbatch = pos.shape[0]
        Nelec = mol.nelec
        # create meshgrid for indexing of electron i and j
        [i, j] = torch.meshgrid(torch.arange(
            0, Nelec), torch.arange(0,Nelec))
        xij = torch.zeros((Nbatch, Nelec, Nelec, 3))
        
        # determine electron - electron distance vector
        xij[:, :, :,:] = (pos[:, i, :]-pos[:, j, :]).reshape(
            Nbatch, Nelec, Nelec, 3)
        
        # determine absolute distance
        rij = torch.norm(xij, dim=3)
        h_0_ij = torch.cat(
            (xij, rij.reshape(Nbatch, Nelec, Nelec, 1)), axis=3)
        # output h_0_ij Nbatch x Nelec x Nelec x Ndim+1
        return h_0_ij



