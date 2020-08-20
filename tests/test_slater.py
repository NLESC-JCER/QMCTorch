import torch
import torch.optim as optim

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.utils import InterpolateAtomicOribtals, InterpolateMolecularOrbitals
from qmctorch.utils import set_torch_double_precision
import platform

import numpy as np
import unittest


class TestSlater(unittest.TestCase):

    def setUp(self):

        set_torch_double_precision()

        self.mol = Molecule(atom='C 0 0 0; O 0 0 2.173; O 0 0 -2.173',
                            calculator='pyscf',
                            basis='dzp',
                            unit='bohr')

        self.wf = Orbital(self.mol, kinetic='jacobi',
                          configs='single_double(6,6)',
                          use_jastrow=True,
                          include_all_mo=False)

        self.wf_allmo = Orbital(self.mol, kinetic='jacobi',
                                configs='single_double(6,6)',
                                use_jastrow=True,
                                include_all_mo=True)

        self.random_fc_weight = torch.rand(self.wf.fc.weight.shape)
        self.wf.fc.weight.data = self.random_fc_weight
        self.wf_allmo.fc.weight.data = self.random_fc_weight

    def test_det(self):

        mo = torch.rand(10, 22, 45)
        det_explicit = self.wf.pool.det_explicit(mo)
        det_single = self.wf.pool.det_single_double(mo)
        assert(torch.allclose(det_explicit, det_single))

    def test_det_all_mo(self):

        mo = torch.rand(10, 22, 45)
        det_explicit = self.wf_allmo.pool.det_explicit(mo)
        det_single = self.wf_allmo.pool.det_single_double(mo)
        assert(torch.allclose(det_explicit, det_single))

    def test_op(self):

        mo = torch.rand(10, 22, 45)
        bkin = torch.rand(10, 22, 45)
        kin_explicit = self.wf.pool.operator_explicit(mo, bkin)
        kin = self.wf.pool.operator_single_double(mo, bkin)
        assert(torch.allclose(kin_explicit, kin))

    def test_op_all_mo(self):

        mo = torch.rand(10, 22, 45)
        bkin = torch.rand(10, 22, 45)
        kin_explicit = self.wf_allmo.pool.operator_explicit(mo, bkin)
        kin = self.wf_allmo.pool.operator_single_double(mo, bkin)
        assert(torch.allclose(kin_explicit, kin))

    def test_multiple_ops(self):

        mo = torch.rand(10, 22, 45)
        bop = torch.rand(6, 10, 22, 45)
        op_explicit = self.wf_allmo.pool.operator_explicit(mo, bop)
        op = self.wf_allmo.pool.operator_single_double(mo, bop)
        assert(torch.allclose(op_explicit, op))


if __name__ == "__main__":
    unittest.main()
    # t = TestSlater()
    # t.setUp()
    # t.test_det()
    # t.test_det_all_mo()
    # t.test_op()
    # t.test_op_all_mo()
    # t.test_multiple_ops()
