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
from qmctorch import log


class FermiPrepocess(nn.Module):

    def __init__(self, mol):
        """Preprocessing of the electron position to the input of the first layer.
        Args:
            mol (Molcule instance): Molecule object
        """
        super(FermiPrepocess,self).__init__()
        self.mol = mol

    def forward(self, pos):
        """Preprocesses the electron positions for input into the FermiNet.
            the distance between electron and nuclei and electron electron are computed
       
        Args:
            pos (torch.tensor): sampling points (Nbatch, Nelec, Ndim)
  
        Returns:
            h_i (torch.tensor): electron-nuclei distances for one-electron stream (Nbatch, Nelec, Natom*(Ndim+1))
            h_ij (torch.tensor): electron-electron distances for two electron stream (Nbatch, Nelec, Nelec, Ndim+1)
        """

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
            pos : electron positions (Nbatch, Nelec, Ndim)
            mol (object) : Molecule instance
        
        Returns:
            h_0_i (torch.tensor): concatenated matrix for the one-electron stream (Nbatch, Nelec, (3*Natom+2)*(Ndim+1))

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
            pos : electron positions (Nbatch, Nelec, Ndim)
            mol (object) : Molecule instance

        Returns:
            h_0_ij (torch.tensor): electron-electron distances for the two electron stream (Nbatch, Nelec, Nelec, Ndim+1)  
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
        """Implementatation of a single intermediate layer of the FermiNet.

        Args:
            mol (qmc.wavefunction.Molecule): a molecule object
            input_size_e (int): input size of the one electron stream
            output_size_e (int): output size of the one electron stream
            input_size_ee (int): input size of the two electron stream
            output_size_ee (int): output size of the two electron stream             
        """

        super(FermiIntermediate, self).__init__()
        self.mol = mol
        self.lin_layer_e = nn.Linear(
            self.f_size(input_size_e,input_size_ee), output_size_e, bias=True)
        self.lin_layer_ee = nn.Linear(
            input_size_ee, output_size_ee, bias=True)

    def forward(self, h_i, h_ij):
        """Computes the output of a sinlge intermediate layer of the FermiNet.
            The one-electron stream input is concatenated with 
       
        Args:
            h_i (torch.tensor): one-electron stream (Nbatch, Nelec, input_size_e)
            h_ij (torch.tensor): two electron stream (Nbatch, Nelec, Nelec, input_size_ee)
  
        Returns:
            h_i (torch.tensor): one-electron stream (Nbatch, Nelec, output_size_e)
            h_ij (torch.tensor): two electron stream (Nbatch, Nelec, Nelec, output_size_ee)         
        """
        # for the one electron stream:\
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
            h_i (torch.tensor): one electron stream output of previous layer Nbatch x Nelec x input_size_e
            h_ij (torch.tensor): two electron stream output of previous layer Nbatch x Nelec x Nelec x input_size_ee
            mol : Molecule instance 
        
        Returns:
            f_i (torch.tensor): (N_batch, Nelec, 3*input_size_e + 2*(inpute_size_ee))
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

class AllOrbitals(nn.Module):

    def __init__(self, nelec, atom_coords, Kdet, input_size):
        """Computes all the orbitals from the output of the last intermediate layer of the FermiNet.
         
        Args:
            nelec (int): number of electrons for the mo matrix
            atom_coords (torch.tensor): atomic coordinates (Natom Ndim)
            Kdet (int): Number of determinants
            input_size (int): input size of the one electron stream
        """
        super(AllOrbitals, self).__init__()

        self.nelec = nelec
        self.atom_coords = atom_coords
        self.natom = len(atom_coords)
        self.Kdet = Kdet
        self.ndim = 3

        self.W = nn.Parameter(torch.rand(
            self.Kdet, input_size, self.nelec))

        self.G = nn.Parameter(torch.rand(
            self.Kdet, self.nelec))

        self.Sigma = nn.Parameter(torch.rand(
            self.Kdet, self.nelec, self.natom, self.ndim, self.ndim))

        self.Pi = nn.Parameter(
            torch.rand(self.Kdet, self.nelec, self.natom))

    def forward(self, h_i, r_i):
        """ Computes the output of all the orbitals as a mo slater matrix
       
        .. math::
            \\phi^{ka}_{i}(r_i, h_i) = (w_i^{ka}.h_j^{La} + g_i^{ka}) \\sum_m\\pi^{ka}_{im} \\exp(-|\\Sigma^{ka}_{im}(r^{a}_j-R_m)|)

        Args:
            h_i (torch.tensor): one-electron stream (Nbatch, Nelec, input_size)
            r_i (torch.tensor): electron positions (Nbatch, Nelec, Ndim)
  
        Returns:
            torch.tensor: mo matrix of orbitals (Nbatch, Ndet, Nelec, Nelec)      

        """
        self.nbatch = r_i.shape[0]

        Rnuclei = torch.tensor(self.atom_coords)

        # compute the orbital part
        # -> Nbatch, Ndet, Norb, Nelec
        out = h_i.unsqueeze(1) @ self.W.unsqueeze(0) 
        out = out.transpose(2,3)
        out = out+ self.G.unsqueeze(2)

        # position of each elec wrt each atom
        # Nbatch, Nelec, Natom, 3
        xyz = (r_i.view(-1, self.nelec, 1, 3) - Rnuclei[None, ...])

        
        # sigma x (r-R)
        # Nbatch, Ndet, Norb, Nelec, Natom, Ndim, 1
        x = xyz.unsqueeze(1).unsqueeze(2).unsqueeze(-2) @ self.Sigma.unsqueeze(0).unsqueeze(3)

        # norm of the vector
        # Nbatch, Ndet, Norb, Nelec, Natom
        x = torch.norm(x.squeeze(-2), dim=-1)

        # exponential
        # Nbatch, Ndet, Norb, Nelec, Natom
        x = torch.exp(-x)
  
        # multiply with pi (pi: Ndet Nelec Natom)
        # Nbatch, Ndet, Norb, Nelec, Natom
        x = self.Pi.unsqueeze(0).unsqueeze(-2) * x

        # sum over the atoms
        # Nbatch, Ndet, Norb, Nelec 
        x = x.sum(-1)

        return out * x


class FermiNet(WaveFunction):

    def __init__(self, mol, hidden_nodes_e=256, hidden_nodes_ee=32, L_layers=4, Kdet=1, kinetic="auto", cuda=False):

        """Wave function FermiNet as proposed by Foulkes and Deepmind [arXiv:1909.02487v2].
         
        Args:
            mol (qmc.wavefunction.Molecule): a molecule object
            input_size_e (int): input size of the one electron stream
            output_size_e (int): output size of the one electron stream
            input_size_ee (int): input size of the two electron stream
            output_size_ee (int): output size of the two electron stream             
            Kdet (int): Number of determinants
            kinetic (str, optional): method to compute the kinetic energy. Defaults to 'auto'.
            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
        """
        super(FermiNet, self).__init__(mol.nelec, 3, kinetic, cuda)
       
        self.mol= mol
        self.ndim = 3
        self.natom = mol.natom
        self.atom_coords = mol.atom_coords

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
                
        # orbital up layer
        self.orb_up = AllOrbitals(
            mol.nup, mol.atom_coords, Kdet, self.hidden_nodes_e)

        # orbital down layer
        self.orb_down = AllOrbitals(
            mol.ndown, mol.atom_coords, Kdet, self.hidden_nodes_e)

        # ci sum
        self.weighted_sum = nn.Linear(self.Kdet,1,bias=False)
        self.log_data()
    
    def log_data(self):
        log.info('')
        log.info(' Wave Function FermiNet')
        log.info('  Intermediate layers : {0}', self.L_layers)
        log.info('  One electron nodes  : {0}', self.hidden_nodes_e)
        log.info('  Two electron nodes  : {0}', self.hidden_nodes_ee)
        log.info('  Number of det       : {0}', self.Kdet)
        log.info('  Number var  param   : {0}', self.get_number_parameters())
        log.info('  Cuda support        : {0}', self.cuda)

    def compute_mo(self, pos):
        """ Computes the output of the FermiNet until the mo slater matrices.   

        Args:
            pos (torch.tensor): electron positions (Nbatch, Nelec, Ndim)
  
        Returns:
            mo_up (torch.tensor): mo matrix of orbitals for spin up (Nbatch, Ndet, Nup, Nup)   
            mo_down (torch.tensor): mo matrix of orbitals for spin up (Nbatch, Ndet, Ndown, Ndown)    

        """
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
        
        # comute the orbital up/down from the output of the last
        # intermediate layer
        mo_up = self.orb_up(h_i[:, :self.mol.nup, :],
                           pos[:, :self.mol.nup, :])
        mo_down = self.orb_down(
            h_i[:, self.mol.nup:, :], pos[:, self.mol.nup:, :])

        return mo_up, mo_down


    def forward(self, pos):
        """ Computes the output of the FermiNet.

        Args:
            pos (torch.tensor): electron positions (Nbatch, Nelec, Ndim)
  
        Returns:
            torch.tensor : values of the wave functions at each sampling point (Nbatch, 1)   

        """
        mo_up,mo_down = self.compute_mo(pos)
        # compute the different slater determinant from the up and down mo slater matrices
        slat_det = torch.det(mo_up)*torch.det(mo_down)
        # compute a weighted sum of the slater determinants.
        return self.weighted_sum(slat_det)
    
    def forward_log(self, pos):
        """ Computes the log of the absolute of the output of the FermiNet.

        Args:
            pos (torch.tensor): electron positions (Nbatch, Nelec, Ndim)
  
        Returns:
            torch.tensor : logarithm of the values of the wave functions at each sampling point (Nbatch, 1)  
            sign (torch.tensor): sign of the wave function (Nbatch, 1) 

        """
        # compute log output of the network
        psi = self.forward(pos)
        return torch.log(torch.abs(psi)), torch.sign(psi)
    

    



    
