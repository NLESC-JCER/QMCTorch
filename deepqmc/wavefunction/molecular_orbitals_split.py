import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

import numpy as np
from pyscf import scf, gto, mcscf

from tqdm import tqdm
from time import time

class MolecularOrbitals(nn.Module):

    """Applies a slater determinant pooling in the active space."""

    def __init__(self,mo_coeffs,configs,nup,ndown):

        super(MolecularOrbitals, self).__init__()

        self.mo_coeffs = mo_coeffs
        self.nmo = self.mo_coeffs.shape[0]

        self.configs = configs
        self.nconfs = len(configs[0])
        
        self.index_up = torch.arange(nup)
        self.index_down = torch.arange(nup,nup+ndown)

        assert(nup==ndown)
        self.W = torch.zeros(self.nconfs,self.nmo,self.nup)

    def _get_matrix(self):
           
        for ic,(cup,cdown) in enumerate(zip(self.configs[0],self.configs[1])):

            W[2*ic] = self.mo_coeffs.index_select(1,cup)
            W[2*ic+1] = self.mo_coeffs.index_select(1,cdown)
            
if __name__ == "__main__":


    x = Variable(torch.rand(10,5,5))
    x.requires_grad = True
    det = BatchDeterminant.apply(x)
    det.backward(torch.ones(10))

    det_true = torch.tensor([torch.det(xi).item() for xi in x])
    print(det-det_true)
