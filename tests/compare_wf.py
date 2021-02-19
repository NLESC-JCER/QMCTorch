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


torch.manual_seed(101)
np.random.seed(101)

set_torch_double_precision()

# molecule
mol = Molecule(
    atom='Li 0 0 0; H 0 0 3.015',
    unit='bohr',
    calculator='pyscf',
    basis='sto-3g',
    redo_scf=True)

# wf
wfc = CorrelatedOrbital(
    mol,
    kinetic='auto',
    jastrow_type='pade_jastrow',
    configs='single_double(2,4)',
    include_all_mo=True)


# wf
wf = Orbital(
    mol,
    kinetic='auto',
    jastrow_type='pade_jastrow',
    configs='single_double(2,4)',
    include_all_mo=True)

random_fc_weight = torch.rand(wf.fc.weight.shape)
wf.fc.weight.data = random_fc_weight
wfc.fc.weight.data = random_fc_weight

nbatch = 3
pos = torch.tensor(np.random.rand(nbatch, wf.nelec*3))
pos.requires_grad = True


print(wf(pos))
print(wfc(pos))
