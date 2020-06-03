import torch
from torch import nn

from .atomic_orbitals import AtomicOrbitals

from .wf_base import WaveFunction
from .intermediate_FermiNet import IntermediateLayers
from .orbital_FermiNet import Orbital_FermiNet

from ..utils import register_extra_attributes
from ..utils.interpolate import (get_reg_grid, get_log_grid,
                                 interpolator_reg_grid, interpolate_reg_grid,
                                 interpolator_irreg_grid, interpolate_irreg_grid)
from qmctorch.utils import timeit

class FermiNet(WaveFunction)

    def __init__(self, mol, hidden_nodes_e=256, hidden_nodes_ee=32, L_layers=4, K_determinants=1, configs='ground_state',
                 kinetic='jacobi', cuda=False)
        Implementation of the QMC Network Fermi Net.

        Args
            mol (qmc.wavefunction.Molecule) a molecule object 
            configs (str, optional) defines the CI configurations to be used. Defaults to 'ground_state'.
            kinetic (str, optional) method to compute the kinetic energy. Defaults to 'jacobi'.
            use_jastrow (bool, optional) turn jastrow factor ONOFF. Defaults to True.
            cuda (bool, optional) turns GPU ONOFF  Defaults to False.
        
        super(FermiNet, self).__init__(mol.nelec, 3, kinetic, cuda)

        # check for cuda
        if not torch.cuda.is_available and self.wf.cuda
            raise ValueError('Cuda not available, use cuda=False')

        # number of atoms
        self.mol = mol
        self.atoms = mol.atoms
        self.natom = mol.natom
        self.nelec = mol.nelec
        self.ndim =3
        
        # hyperparameters of network
        self.hidden_nodes_e = hidden_nodes_e
        self.hidden_nodes_ee = hidden_nodes_ee
        self.K_determinants = K_determinants
        self.L_layers = L_layers
        
       # Create the intermediate layer network
        self.Intermediate_layer = IntermediateLayers(
            self.mol, self.hidden_nodes_e, self.hidden_nodes_ee, self.L_layers, self.ndim, cuda)
        # create KN_up spin up orbitals and KN_down spin down orbitals with K the number of determinants.
        self.Orbital_determinant_up = nn.ModuleList()
        self.Orbital_determinant_down = nn.ModuleList()
        for k in range(self.K_determinants)
            Orbitals_up = nn.ModuleList()
            Orbitals_down = nn.ModuleList()
            for i in range(self.mol.nup)
                Orbitals_up.append(Orbital_FermiNet(self.mol, hidden_nodes_e))

            for i in range(self.mol.ndown)
                Orbitals_down.append(Orbital_FermiNet(self.mol, hidden_nodes_e))

            self.Orbital_determinant_up.append(Orbitals_up)
            self.Orbital_determinant_down.append(Orbitals_down)
        # Create a final linear layer of weighted determinants.
        self.weighted_sum = nn.Linear(K_determinants, 1, bias=False)


    def forward(self, pos)

        '''
        forward electron position through the Fermi Net. 
        First through intermediate layers then create seperate orbitals 
        which are combined in a slater determinant.

        Args 
            pos (float) electron position Nbatch x [Nelec x Ndim]      
        
        '''

        pos = pos.view(pos.shape[0],self.mol.nelec,self.ndim)
        self.nbatch = pos.shape[0]

        # Go through the intermediate layers 
        h_L_i, h_L_ij = self.Intermediate_layer(pos)
        # Create the determinants 
        determinant = torch.zeros(self.nbatch, self.K_determinants)
        for k in range(self.K_determinants)
            # for each determinant k create two slatter determinants one spin-up one spin-down
            det_up = torch.zeros((self.nbatch, self.mol.nup, self.mol.nup))
            det_down = torch.zeros((self.nbatch, self.mol.ndown, self.mol.ndown))
            # for up
            for i_up in range(self.mol.nup)
                det_up[, i_up, ] = self.Orbital_determinant_up[k][i_up](
                    h_L_i[,self.mol.nup].reshape((self.nbatchself.mol.nup,self.hidden_nodes_e)),
                    pos[,self.mol.nup].reshape((self.nbatchself.mol.nup,self.ndim))).reshape((self.nbatch,self.mol.nup))
            # for down        
            for i_down in range(self.mol.ndown)
                det_down[, i_down,] = self.Orbital_determinant_up[k][i_down](
                    h_L_i[,self.mol.nup].reshape((self.nbatchself.mol.ndown,self.hidden_nodes_e)),
                    pos[,self.mol.nup].reshape((self.nbatchself.mol.ndown,self.ndim))).reshape((self.nbatch,self.mol.ndown))
            determinant[, k] = torch.det(det_up)torch.det(det_down)
        #create make a weighted sum of the determinants
        psi = self.weighted_sum(determinant)
        # return psi.
        # in the optimization log(psi) is used as output of the network
        # return torch.log(psi)
        return psi       

    def nuclear_potential(self, pos)
        Computes the electron-nuclear term

        .. math
            V_{en} = - sum_e sum_n frac{Z_n}{r_{en}}

        Args
            x (torch.tensor) sampling points (Nbatch, 3Nelec)

        Returns
            torch.tensor values of the electon-nuclear energy at each sampling points
        

        p = torch.zeros(pos.shape[0], device=self.device)
        for ielec in range(self.nelec)
            istart = ielec  self.ndim
            iend = (ielec + 1)  self.ndim
            pelec = pos[, istartiend]
            for iatom in range(self.natom)
                patom = self.ao.atom_coords[iatom, ]
                Z = self.ao.atomic_number[iatom]
                r = torch.sqrt(((pelec - patom)2).sum(1))  # + 1E-12
                p += -Z  r
        return p.view(-1, 1)

    def electronic_potential(self, pos)
        Computes the electron-electron term

        .. math
            V_{ee} = sum_{e_1} sum_{e_2} frac{1}{r_{e_1e_2}}

        Args
            x (torch.tensor) sampling points (Nbatch, 3Nelec)

        Returns
            torch.tensor values of the electon-electron energy at each sampling points
        

        pot = torch.zeros(pos.shape[0], device=self.device)

        for ielec1 in range(self.nelec - 1)
            epos1 = pos[, ielec1 
                        self.ndim(ielec1 + 1)  self.ndim]
            for ielec2 in range(ielec1 + 1, self.nelec)
                epos2 = pos[, ielec2 
                            self.ndim(ielec2 + 1)  self.ndim]
                r = torch.sqrt(((epos1 - epos2)2).sum(1))  # + 1E-12
                pot += (1.  r)
        return pot.view(-1, 1)

    def nuclear_repulsion(self)
        Computes the nuclear-nuclear repulsion term

        .. math
            V_{nn} = sum_{n_1} sum_{n_2} frac{Z_1Z_2}{r_{n_1n_2}}

        Returns
            torch.tensor values of the nuclear-nuclear energy at each sampling points
        

        vnn = 0.
        for at1 in range(self.natom - 1)
            c0 = self.ao.atom_coords[at1, ]
            Z0 = self.ao.atomic_number[at1]
            for at2 in range(at1 + 1, self.natom)
                c1 = self.ao.atom_coords[at2, ]
                Z1 = self.ao.atomic_number[at2]
                rnn = torch.sqrt(((c0 - c1)2).sum())
                vnn += Z0  Z1  rnn
        return vnn

    def geometry(self, pos)
        Returns the gemoetry of the system in xyz format

        Args
            pos (torch.tensor) sampling points (Nbatch, 3Nelec)

        Returns
            list list where each element is one line of the xyz file
        
        d = []
        for iat in range(self.natom)
            at = self.atoms[iat]
            xyz = self.ao.atom_coords[iat,
                                      ].detach().numpy().tolist()
            d.append((at, xyz))
        return d

