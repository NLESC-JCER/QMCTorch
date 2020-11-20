from qmctorch.scf import Molecule
from qmctorch.wavefunction import CorrelatedOrbital
from qmctorch.wavefunction import Orbital
from qmctorch.utils import set_torch_double_precision

from torch.autograd import grad, gradcheck, Variable

import numpy as np
import torch
import unittest
import itertools
import os


torch.set_default_tensor_type(torch.DoubleTensor)

# molecule
mol = Molecule(
    atom='H 0 0 0; H 0 0 1.',
    unit='bohr',
    calculator='pyscf',
    basis='dz',
    redo_scf=True)

wf = CorrelatedOrbital(
    mol, kinetic='auto', configs='single_double(2,2)')

# self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
# self.wf.fc.weight.data = self.random_fc_weight

nbatch = 10
pos = torch.tensor(np.random.rand(
    nbatch, wf.nelec*3))
pos.requires_grad = True
