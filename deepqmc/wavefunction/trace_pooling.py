import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

import numpy as np
from pyscf import scf, gto, mcscf

from tqdm import tqdm
from time import time

class TracePooling(nn.Module):

    """Applies a slater determinant pooling in the active space."""

    def __init__(self,configs,nup,ndown):
        super(TracePooling, self).__init__()

    def forward(self,mo_up,mo_down):

        """Compute the trace of the different MO matrices
        and mulitply spin up/down for each confs.

        Args:
            mo_up (torch.tensor) : mo of spin ups
            mo_down (torch.tensor) : mo of spin down

        Returns:
            torch.tensor: Values of the SDs
        """

       return torch.det(mo_up) * torch.det(mo_down)
            
