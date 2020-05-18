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

    def test_kin(self):

        mo = torch.rand(10, 22, 45)
        bkin = torch.rand(10, 22, 45)
        kin_explicit = self.wf.pool.kinetic_explicit(mo, bkin)
        kin = self.wf.pool.kinetic_single_double(mo, bkin)
        assert(torch.allclose(kin_explicit, kin))

    def test_kin_all_mo(self):

        mo = torch.rand(10, 22, 45)
        bkin = torch.rand(10, 22, 45)
        kin_explicit = self.wf_allmo.pool.kinetic_explicit(mo, bkin)
        kin = self.wf_allmo.pool.kinetic_single_double(mo, bkin)
        assert(torch.allclose(kin_explicit, kin))


if __name__ == "__main__":
    # unittest.main()
    t = TestSlater()
    t.setUp()
    t.test_det()
    t.test_det_all_mo()
    t.test_kin()
    t.test_kin_all_mo()
