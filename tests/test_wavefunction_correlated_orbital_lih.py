from qmctorch.scf import Molecule
from qmctorch.wavefunction import CorrelatedOrbital
from qmctorch.wavefunction import Orbital
from qmctorch.utils import set_torch_double_precision, btrace

from base_test_wavefunction_correlated_orbital import BaseTestCorrelatedOrbitalWF
from base_test_wavefunction_correlated_orbital import hess

from torch.autograd import grad, gradcheck, Variable

import numpy as np
import torch
import unittest
import itertools
import os


import operator


torch.set_default_tensor_type(torch.DoubleTensor)


class TestCorrelatedOrbitalWF(BaseTestCorrelatedOrbitalWF):

    def setUp(self):

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

        self.wf = CorrelatedOrbital(
            mol,
            kinetic='auto',
            jastrow_type='pade_jastrow',
            configs='ground_state',
            include_all_mo=True)

        # self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        # self.wf.fc.weight.data = self.random_fc_weight

        self.nbatch = 10
        self.pos = torch.tensor(np.random.rand(
            self.nbatch, self.wf.nelec*3))

        self.pos.requires_grad = True

    def test_forward(self):
        """Value of the wave function."""
        wfvals = self.wf(self.pos)
        ref = torch.tensor([[-1.0935e-02], [6.4874e-02], [1.7879e-04],
                            [1.5797e-02], [7.4684e-02], [-4.4445e-02],
                            [-4.8149e-04], [-3.0355e-03], [-2.0027e-02],
                            [5.1957e-05]])

        # assert torch.allclose(wfvals.data, ref, rtol=1E-4, atol=1E-4)


if __name__ == "__main__":

    set_torch_double_precision()

    t = TestCorrelatedOrbitalWF()
    t.setUp()
    t.test_forward()

    t.test_jacobian_mo()
    t.test_grad_mo()
    t.test_hess_mo()

    t.test_jacobian_jast()
    t.test_grad_jast()
    t.test_hess_jast()

    t.test_grad_cmo()
    t.test_hess_cmo()

    t.test_grad_manual()
    t.test_hess_manual()

    t.test_jacobian_wf()
    t.test_grad_wf()

    t.test_grad_slater_det()
    t.test_hess_slater_det_manual()
    t.test_hess_slater_det()

    # t.test_kinetic_energy()
    # t.test_local_energy()
