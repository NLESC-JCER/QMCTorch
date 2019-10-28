import sys
import numpy as np
import torch
from torch import nn


from deepqmc.wavefunction.wf_base import WaveFunction
from deepqmc.wavefunction.atomic_orbitals import AtomicOrbitals
from deepqmc.wavefunction.slater_pooling import SlaterPooling
from deepqmc.wavefunction.jastrow import TwoBodyJastrowFactor, ElectronDistance



class Orbital(WaveFunction):

    def __init__(self,mol,configs='ground_state',scf='pyscf'):
        super(Orbital,self).__init__(mol.nelec,3)

        # number of atoms
        self.atoms = mol.atoms
        self.bonds = mol.bonds
        self.natom = mol.natom

        # define the atomic orbital layer
        self.ao = AtomicOrbitals(mol)

        # define the mo layer
        self.mo = nn.Linear(mol.norb, mol.norb, bias=False)

        # initialize the MO coefficients
        mo_coeff =  torch.tensor(mol.get_mo_coeffs(code=scf)).float()
        self.mo.weight = nn.Parameter(mo_coeff.transpose(0,1))

        # jastrow
        self.edist = ElectronDistance(mol.nelec,3)
        self.jastrow = TwoBodyJastrowFactor(mol.nup,mol.ndown)

        # define the SD we want
        self.configs = self.get_configs(configs,mol)
        #self.configs = (torch.LongTensor([np.array([0])]), torch.LongTensor([np.array([0])]))
        self.nci = len(self.configs[0])

        #  define the SD pooling layer
        self.pool = SlaterPooling(self.configs,mol.nup,mol.ndown)

        # define the linear layer
        self.fc = nn.Linear(self.nci, 1, bias=False)
        self.fc.weight.data.fill_(1.)
        self.fc.clip = False
        
    def forward(self,x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''
        
        edist  = self.edist(x)
        J = self.jastrow(edist)

        x = self.ao(x)
        x = self.mo(x)
        x = self.pool(x)

        return J*x

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi

        TODO : vecorize that !! 
        '''
        
        p = torch.zeros(pos.shape[0])
        for ielec in range(self.nelec):
            pelec = pos[:,(ielec*self.ndim):(ielec+1)*self.ndim]
            for iatom in range(self.natom):
                patom = self.ao.atom_coords[iatom,:]
                Z = self.ao.atomic_number[iatom]
                r = torch.sqrt(   ((pelec-patom)**2).sum(1)  ) + 1E-6
                p += (-Z/r)
        return p.view(-1,1)

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''

        pot = torch.zeros(pos.shape[0])
        
        for ielec1 in range(self.nelec-1):
            epos1 = pos[:,ielec1*self.ndim:(ielec1+1)*self.ndim]
            
            for ielec2 in range(ielec1+1,self.nelec):
                epos2 = pos[:,ielec2*self.ndim:(ielec2+1)*self.ndim]
                
                r = torch.sqrt( ((epos1-epos2)**2).sum(1) ) + 1E-6
                pot = (1./r) 

        return pot.view(-1,1)

    def nuclear_repulsion(self):
        '''Compute the nuclear repulsion term    
        Returns: values of Vnn
        '''

        rnn = 0.
        for at1 in range(self.natom-1):
            c0 = self.ao.atom_coords[at1,:]
            Z0 = self.ao.atomic_number[at1]
            for at2 in range(at1+1,self.natom):
                c1 = self.ao.atom_coords[at2,:]
                Z1 = self.ao.atomic_number[at2]
                rnn += torch.sqrt(   ((c0-c1)**2).sum()  )
        return (Z0*Z1/rnn).view(-1,1)


    def atomic_distances(self,pos):
        d = []
        for iat1 in range(self.natom-1):
            at1 = self.atoms[iat1]
            c1 = self.ao.atom_coords[iat1,:]
            for iat2 in range(iat1+1,self.natom):
                at2 = self.atoms[iat2]
                c2 = self.ao.atom_coords[iat2,:]
                d.append((at1,at2,torch.sqrt(   ((c1-c2)**2).sum())))
        return d

    def get_configs(self,configs,mol):

        if isinstance(configs,torch.Tensor):
            return configs

        elif configs == 'ground_state':
            return self._get_ground_state_config(mol)

    def _get_ground_state_config(self,mol):
        conf = (torch.LongTensor([np.array(range(mol.nup))]), 
                torch.LongTensor([np.array(range(mol.ndown))]))
        return conf











