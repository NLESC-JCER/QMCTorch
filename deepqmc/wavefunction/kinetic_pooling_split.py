import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

import numpy as np
from pyscf import scf, gto, mcscf

from tqdm import tqdm
from time import time

def btrace(M):
    return torch.diagonal(M,dim1=-2,dim2=-1).sum(-1)

class KineticPooling(nn.Module):

    """Comutes the kinetic energy of each configuration using the trace trick."""

    def __init__(self):
        super(KineticPooling, self).__init__()

    def forward(self,MO, d2MO, return_local_energy=False):

        ''' Compute the kinetic energy using the trace trick
        for a product of spin up/down determinant
        .. math::

            T \Psi  =  T Dup Ddwn 
                    = -1/2 Dup * Ddown  *( \Delta_up Dup  + \Delta_down Ddwn)

            using the trace trick with D = |A| :
                O(D) = D trace(A^{-1} O(A))
                and Delta_up(D_down) = 0

        Args:
            A : matrix of MO vals (Nbatch, Nelec, Nmo)
            d2A : matrix of \Delta MO vals (Nbatch, Nelec, Nmo)
        Return:
            K : T Psi (Nbatch, Ndet)
        '''

        # shortcut up/down matrices
        Aup, Adown = MO
        d2Aup, d2Adown = d2MO
        
        # size
        nbatch = Aup.shape[0]
        nconfs = Aup.shape[1]

        # inverse of MO matrices        
        iAup = torch.inverse(Aup)
        iAdown = torch.inverse(Adown)

        # product
        out = (btrace(iAup@d2Aup) + btrace(iAdown@d2Adown))

        # multiply by det if necessary
        if not return_local_energy:
            pd = torch.det(Aup) * torch.det(Adown)
            out[:,ic] *= pd

        return -0.5*out.view(-1,1)
