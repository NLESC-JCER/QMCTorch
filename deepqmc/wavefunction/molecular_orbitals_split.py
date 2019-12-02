import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

import numpy as np
from pyscf import scf, gto, mcscf

from tqdm import tqdm
from time import time

class MolecularOrbitals(nn.Module):

    """Computes the MO from the AO a la Ferminet."""

    def __init__(self,mo_coeffs,configs,nup,ndown):

        super(MolecularOrbitals, self).__init__()

        self.mo_coeffs = mo_coeffs
        self.nmo = self.mo_coeffs.shape[0]

        self.configs = configs
        self.nconfs = len(configs[0])
        
        self.index_up = torch.arange(nup)
        self.index_down = torch.arange(nup,nup+ndown)

        assert(nup==ndown)
        self.nup = nup
        self.ndown = ndown

        self.Wup = torch.zeros(self.nconfs,self.nmo,self.nup)
        self.Wdown = torch.zeros(self.nconfs,self.nmo,self.nup)

        self._get_matrices()

    def _get_matrices(self):
        '''Computes the matrix W that expand the AO in the MOs of each
        spin up / spin down configuration.'''
        for ic,(cup,cdown) in enumerate(zip(self.configs[0],self.configs[1])):

            self.Wup[ic] = self.mo_coeffs.index_select(1,cup)
            self.Wdown[ic] = self.mo_coeffs.index_select(1,cdown)

    def forward(self,ao):
        """compute the MOs.
        
        Args:
            ao (torch.tensor): values of the atomic orbitals
        
        Returns:
            torch.tensor: values of the MO in each configuration
        """
        
        return self._project(ao[:,:self.nup,:],self.Wup), \
            self._project(ao[:,self.nup:,:],self.Wdown)

    def _project(self,ao,w):
        return ao.unsqueeze(1) @ w.unsqueeze(0)

            