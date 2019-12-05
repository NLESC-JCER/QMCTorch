import sys
import numpy as np
import torch
from torch import nn
from time import time

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.wavefunction.atomic_orbitals import AtomicOrbitals

from deepqmc.wavefunction.kinetic_pooling_split import KineticPooling
from deepqmc.wavefunction.jastrow import TwoBodyJastrowFactor, ElectronDistance

from deepqmc.wavefunction.trace_pooling import TracePooling

class OrbitalSplit(Orbital):

    def __init__(self,mol,configs='ground_state',scf='pyscf',kinetic_jacobi=False):
        super(OrbitalSplit,self).__init__(mol,configs,scf,kinetic_jacobi)

        # define the SD we want
        self.configs = self.get_configs(configs,mol)
        self.nconfs = len(configs[0])

        # get the projectors of the confs
        self.Pup, self.Pdown = self.get_projectors()

        #  define the SD pooling layer
        self.pool = TracePooling()

        # poolin operation to directly compute the kinetic energies via Jacobi formula
        self.kinpool = KineticPooling()


    def get_projectors(self):
        """Get the projectors of the conf in the CI expansion
        
        Returns:
            torch.tensor, torch.tensor : projectors
        """
        nmo = self.mol.nmo
        nup = self.nup
        Pup = torch.zeros(self.nconfs,nmo,nup)
        Pdown = torch.zeros(self.nconfs,nmo,ndown)

        for ic,(cup,cdown) in enumerate(zip(self.configs[0],self.configs[1])):

            for _id,imo in enumerate(cup):
                Pup[ic][_id,imo] = 1.

            for _id,imo in enumerate(cdown):
                Pdown[ic][_id,imo] = 1.

        return Pup.unsqueeze(1), Pdown.unsqueeze(1)

    def split_orbitals(self,mo):
        return mo[:,:self.nup,:] @ Pup, mo[:,self.nup:,:] @ Pdown

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
        mo = self.mo(ao)
        moup, modown = self.split_orbitals(mo)
        sd = self.pool(moup,modown)
        return self.fc(sd)
        #return J*x

    def kinetic_energy_jacobi(self,x,return_local_energy=False, **kwargs):
        '''Compute the value of the kinetic enery using
        the Jacobi formula for derivative of determinant.

        Args:
            x: position of the electrons

        Returns: values of \Delta \Psi
        '''

        MO = self.split_orbitals(self.mo(self.ao(x)))
        d2MO = self.split_orbitals(self.mo(self.ao(x,derivative=2)))
        return self.fc(self.kinpool(MO,d2MO,return_local_energy=return_local_energy))



        











