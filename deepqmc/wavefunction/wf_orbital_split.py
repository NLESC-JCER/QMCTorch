import sys
import numpy as np
import torch
from torch import nn
from time import time

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.wavefunction.atomic_orbitals import AtomicOrbitals
from deepqmc.wavefunction.slater_pooling import SlaterPooling
from deepqmc.wavefunction.kinetic_pooling_split import KineticPooling
from deepqmc.wavefunction.jastrow import TwoBodyJastrowFactor, ElectronDistance

from deepqmc.wavefunction.molecular_orbitals_split import MolecularOrbitals
from deepqmc.wavefunction.trace_pooling import TracePooling

class OrbitalSplit(Orbital):

    def __init__(self,mol,configs='ground_state',scf='pyscf',kinetic_jacobi=False):
        super(OrbitalSplit,self).__init__(mol,configs,scf,kinetic_jacobi)

        # define the SD we want
        self.configs = self.get_configs(configs,mol)
        self.nci = len(self.configs[0])

        # define the mo layer
        self.mo = MolecularOrbitals(self.get_mo_coeffs(),self.configs,mol.nup,mol.ndown)

        #  define the SD pooling layer
        self.pool = TracePooling()

        # poolin operation to directly compute the kinetic energies via Jacobi formula
        self.kinpool = KineticPooling()

    def forward(self,x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            x: position of the electrons

        Returns: values of psi
        '''
        
        #edist  = self.edist(x)
        #J = self.jastrow(edist)

        ao = self.ao(x)
        moup, modown = self.mo(ao)
        sd = self.pool(moup,modown)
        return self.fc(sd)
        #return J*x











