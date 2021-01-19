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


def hess(out, pos):
    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos,
                 grad_outputs=z,
                 only_inputs=True,
                 create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[1]):

        tmp = grad(jacob[:, idim], pos,
                   grad_outputs=z,
                   only_inputs=True,
                   create_graph=True)[0]

        hess[:, idim] = tmp[:, idim]

    return hess


# molecule
mol = Molecule(
    atom='Li 0 0 0; H 0 0 3.015',
    unit='bohr',
    calculator='pyscf',
    basis='sto-3g',
    redo_scf=True)

wf = CorrelatedOrbital(
    mol,
    kinetic='jacobi',
    jastrow_type='pade_jastrow',
    configs='cas(2,2)')

wf.jastrow.weight.data = torch.rand(wf.jastrow.weight.shape)

nbatch = 10
pos = torch.tensor(np.random.rand(
    nbatch, wf.nelec*3))
pos.requires_grad = True


cmo = wf.pos2cmo(pos)

sd = wf.pool(cmo)

bhess = wf.pos2cmo(pos, 2)
hess_vals = wf.pool.operator(cmo, bhess)

bgrad = wf.get_gradient_operator(pos)
grad_vals = wf.pool.operator(cmo, bgrad, op=None)
grad_vals_squared = wf.pool.operator(
    cmo, bgrad, op_squared=True)


hess_jacobi = hess_vals + \
    (grad_vals[0]**2).sum(0) + (grad_vals[1]**2).sum(0) - \
    grad_vals_squared.sum(0) + 2 * operator.mul(*grad_vals).sum(0)

hess_auto = hess(sd, pos).sum(-1).view(-1, 1) / sd
