from qmctorch.scf import Molecule
from qmctorch.wavefunction import CorrelatedOrbital
from qmctorch.wavefunction import Orbital
from qmctorch.utils import set_torch_double_precision, btrace

from torch.autograd import grad, gradcheck, Variable

import numpy as np
import torch
import unittest
import itertools
import os
import operator

torch.set_default_tensor_type(torch.DoubleTensor)

set_torch_double_precision()
torch.manual_seed(101)
np.random.seed(101)

# molecule
mol = Molecule(
    atom='Li 0 0 0; H 0 0 3.015',
    unit='bohr',
    calculator='pyscf',
    basis='sto-3g',
    redo_scf=True)

wf = CorrelatedOrbital(
    mol,
    kinetic='auto',
    jastrow_type='pade_jastrow',
    configs='ground_state')

wf.jastrow.weight.data = torch.rand(wf.jastrow.weight.shape)

nbatch = 10
pos = torch.tensor(np.random.rand(
    nbatch, wf.nelec*3))
pos.requires_grad = True

mo = wf.pos2mo(pos)
dmo = wf.pos2mo(pos, derivative=1, jacobian=False)
d2mo = wf.pos2mo(pos, derivative=2)

jast = wf.ordered_jastrow(pos)
djast = wf.ordered_jastrow(pos, derivative=1, jacobian=False)
d2jast = wf.ordered_jastrow(pos, derivative=2)
