import sys
import numpy as np
import torch
from torch import nn


from deepqmc.wavefunction.wf_base import WaveFunction
from deepqmc.wavefunction.atomic_orbitals import AtomicOrbitals
from deepqmc.wavefunction.slater_pooling import SlaterPooling
from deepqmc.wavefunction.jastrow import TwoBodyJastrowFactor, ElectronDistance



class Orbital(WaveFunction):

    def __init__(self,mol,scf='pyscf'):
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

        # define the SD pooling layer
        self.configs = (torch.LongTensor([np.array([0])]), torch.LongTensor([np.array([0])]))
        self.nci = 1
        self.pool = SlaterPooling(self.configs,1,1)

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
        #x = self.pool(x) #<- issue with batch determinant
        x = (x[:,0,0]*x[:,1,0]).view(-1,1)
        return J*x

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi

        TODO : vecorize that !! The solution below doesn't really wirk :(def plot_observable(obs_dict,e0=None,ax=None):)
        '''
        
        p = torch.zeros(pos.shape[0])
        for ielec in range(self.nelec):
            pelec = pos[:,(ielec*self.ndim):(ielec+1)*self.ndim]
            for iatom in range(self.natom):
                patom = self.ao.atom_coords[iatom,:]
                r = torch.sqrt(   ((pelec-patom)**2).sum(1)  ) + 1E-6
                p += (-1./r)

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
            for at2 in range(at1+1,self.natom):
                c1 = self.ao.atom_coords[at2,:]
                rnn += torch.sqrt(   ((c0-c1)**2).sum()  )
        return (1./rnn).view(-1,1)














