# Fermi Orbital with own Parameter matrices 
import torch 
from torch import nn 

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.solver import SolverOrbital
from qmctorch.sampler import Metropolis
from qmctorch.utils import set_torch_double_precision
from qmctorch.utils import (plot_energy, plot_data)
from qmctorch.wavefunction import WaveFunction



import numpy as np 


class FermiPrepocess(nn.Module):

    def __init__(self, mol):
        """preprocessing of the electron position to the input of the first layer.
        Args:
            mol (Molcule instance): Molecule object
        """
        super(FermiPrepocess,self).__init__()
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

        super(FermiIntermediate, self).__init__()
        self.mol = mol
        self.lin_layer_e = nn.Linear(
            self.f_size(input_size_e,input_size_ee), output_size_e, bias=True)
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

    def __init__(self, mol, Kdet,input_size):
        """Computes all the orbitals from the output of the last layer."""
        super(FermiOrbital, self).__init__()
        self.mol = mol
        self.Kdet = Kdet
        self.ndim = 3

        self.W = nn.Parameter(torch.rand(input_size,self.Kdet )) 
        self.G = nn.Parameter(torch.rand(self.Kdet)) 
        self.Sigma = nn.Parameter(torch.rand(self.Kdet,self.mol.natom,self.ndim,self.ndim))
        self.pi = nn.Parameter(torch.rand(self.Kdet,self.mol.natom))

    def forward(self, h_i, r_i):
        
        self.nbatch = r_i.shape[0]
        
        Rnuclei = torch.tensor(self.mol.atom_coords)
        
        # print("R_m:",Rnuclei.shape)
        # print("r:",r_i.shape)
        # print("W:",self.W.shape)
        # print("G:",self.G.shape)


        # input h_i: nbatch x hidden_nodes_e 
        out = h_i @ self.W  + self.G # outputs shape: nbatch x kdet
        # print("initial linear:",out.shape)
        rim = -Rnuclei[None,:,:] - r_i[:,None,:] #shape: nbatch x natom x ndim

        # print("R_im:",rim.shape)
        # print("Sigma:",self.Sigma.shape)

        expterm = torch.zeros(self.nbatch,self.Kdet,self.mol.natom,self.ndim)

        for m in range(self.mol.natom):
            #input rim: nbatch x 1 x ndim  
            # weight shape: kdet x ndim x ndim 
            expterm[:,:,m,:] = (rim[:,m] @ self.Sigma[:,m]).transpose(0,1) #output shape: nbatch x kdet x 1 x ndim     
        # print("expterm:",expterm.shape)
        # output exterm shape: nbatch x kdet x natom x ndim        
        expnorm = torch.norm(expterm,dim=3)

        # print("expnorm:",expnorm.shape)
        # print("pi:",self.pi.shape)

        # output norm shape: nbatch x kdet x natom        
        Exponential = torch.sum(torch.exp(-expnorm) * self.pi[None,:,:],axis=2)

        # print("Expstuff:",Exponential.shape)


        #output shape: nbatch x kdet
        out = out * Exponential
        return out



class FermiNet(WaveFunction):

    def __init__(self, mol, hidden_nodes_e=256, hidden_nodes_ee=32, L_layers=4, Kdet=1,kinetic="jacobi",cuda=False):

        super(FermiNet, self).__init__(mol.nelec, 3, kinetic, cuda)
       
        self.mol= mol
        self.ndim = 3

        # input network architecutre
        self.hidden_nodes_e = hidden_nodes_e
        self.hidden_nodes_ee = hidden_nodes_ee
        self.L_layers = L_layers
        self.Kdet = Kdet
        
        # preprocess the input positions:
        self.prepoc = FermiPrepocess(mol)

        # create the intermediate layers:
        self.intermediate = nn.ModuleList()
        self.intermediate.append(FermiIntermediate(        
            self.mol,self.mol.natom*(self.ndim+1), self.hidden_nodes_e, 
            self.ndim+1, self.hidden_nodes_ee))
        for l in range(1, self.L_layers):
            self.intermediate.append(FermiIntermediate(
                self.mol,self.hidden_nodes_e, self.hidden_nodes_e,
                self.hidden_nodes_ee, self.hidden_nodes_ee))
                
        self.Orb_up = nn.ModuleList()
        self.Orb_down = nn.ModuleList()
        for i in range(self.mol.nup):
            self.Orb_up.append(FermiOrbital(mol, Kdet, self.hidden_nodes_e))
        for i in range(self.mol.ndown):
            self.Orb_down.append(FermiOrbital(mol,Kdet, self.hidden_nodes_e))
        
        self.weighted_sum = nn.Linear(self.Kdet,1,bias=False)

    def forward_det(self, pos):
        
        
        self.nbatch = pos.shape[0]
        if pos.shape[-1] == self.mol.nelec*self.ndim:
            pos = pos.reshape(self.nbatch,self.mol.nelec,self.ndim)
        elif pos.shape[-1] == self.ndim:
            pos = pos
        else: 
            AttributeError("Input position wrong shape")


        
        # Using these inputs the distance between electron and nuclei and electron electron are computed
        # r_i-R_I, |r_i-R_I| and r_i-r_j, |r_i-r_j|.
        h_i, h_ij = self.prepoc(pos)
        
        # go through multiple linear layers with tanh activation with concatenation for 
        # the one-electron stream of mean of other electron outputs.
        for l in range(self.L_layers):
            h_i,h_ij = self.intermediate[l](h_i,h_ij)
        
        #determine the spin up and spin down determinants
        det_up = torch.zeros((self.nbatch,self.mol.nup,self.mol.nup,self.Kdet))
        det_down = torch.zeros((self.nbatch,self.mol.ndown,self.mol.ndown,self.Kdet))
        # go through the different orbitals
        for i in range(self.mol.nup):
            # reshape the input to directly obtain for all electrons with the given orbital
            # the orbital network directly computes for all K determinants.
            det_up[:,:,i,:] = self.Orb_up[i](h_i[:,:self.mol.nup].reshape((
                self.nbatch*self.mol.nup,self.hidden_nodes_e)),
                            pos[:,:self.mol.nup].reshape((
                                self.nbatch*self.mol.nup,self.ndim))).reshape((
                                    self.nbatch,self.mol.nup,self.Kdet))
        # now compute for spin down
        for i in range(self.mol.ndown):
            det_down[:,:,i,:] = self.Orb_down[i](h_i[:,self.mol.nup:].reshape((
                self.nbatch*self.mol.ndown,self.hidden_nodes_e)),
                            pos[:,self.mol.nup:].reshape((
                                self.nbatch*self.mol.ndown,self.ndim))).reshape((
                                    self.nbatch,self.mol.nup,self.Kdet))
        # transpose the determinants dimension and the dimension over which the electrons are aligned for the slater determinant.
        det_up = det_up.transpose(1,3)
        det_down= det_down.transpose(1,3)
        return det_up, det_down


    def forward(self, pos):
                
        det_up,det_down = self.forward_det(pos)
        # compute the different slater determinant from the up and down determinants
        slat_det = torch.det(det_up)*torch.det(det_down)
        # compute a weighted sum of the slater determinants.
        psi = self.weighted_sum(slat_det)
        return psi
    
    def forward_log(self, pos):
        # compute log output of the network
        return torch.log(torch.abs(self.forward(pos)))
    
def getNumParams(params):
    '''function to get the variable count
    from https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/12'''
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable 


if __name__ == "__main__":
    ndim = 3
    
    
    set_torch_double_precision()
    # define the molecule
    mol = mol = Molecule(atom='O	 0.000000 0.00000  0.00000', 
                unit='bohr', calculator='pyscf')  

 
    # network hyperparameters: 
    hidden_nodes_e = 256
    hidden_nodes_ee = 32
    K_determinants = 1
    L_layers = 4

    # set a initial seed for to make the example reproducable
    torch.random.manual_seed(321)
    nbatch =5
    # initiaite a random configuration of particle positions
    r = torch.randn(nbatch,mol.nelec,ndim, device="cpu")

    # using identical electron positions should if everything is correct return all 0 
    # r = torch.ones((nbatch,mol.nelec, ndim), device="cpu")   

    WF = FermiNet(mol,hidden_nodes_e,hidden_nodes_ee,L_layers,K_determinants)
    # Int = FermiIntermediate(mol,hidden_nodes_e,hidden_nodes_e,hidden_nodes_ee,hidden_nodes_ee)
    # Orb = FermiOrbital(mol,K_determinants,hidden_nodes_e)


    
    # # check the number of parameters and layers of the Network:
    # for name, param in WF.named_parameters():
    #     print(name, param.size())
    # print(getNumParams(WF.parameters()))

    print(WF.forward_log(r))



    
