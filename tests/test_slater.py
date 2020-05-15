import torch
import torch.optim as optim

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.utils import InterpolateAtomicOribtals, InterpolateMolecularOrbitals
import platform

import numpy as np
import unittest


class TestSlater(unittest.TestCase):

    def setUp(self):

        # molecule
        # self.mol = Molecule(
        #     atom='H 0 0 -0.69; H 0 0 0.69',
        #     unit='bohr',
        #     calculator='pyscf',
        #     basis='dzp')

        self.mol = Molecule(atom='C 0 0 0; O 0 0 2.173; O 0 0 -2.173',
                            calculator='pyscf',
                            basis='dzp',
                            unit='bohr')

    def test_single(self):

        wf = Orbital(self.mol, kinetic='jacobi',
                     configs='single(6,6)',
                     use_jastrow=True,
                     include_all_mo=False)

        mo = torch.rand(10, 22, 45)
        det_explicit = wf.pool.det_explicit(mo)
        det_single = wf.pool.det_single(mo)
        assert(torch.allclose(det_explicit, det_single))

    def test_single_all_mo(self):

        wf = Orbital(self.mol, kinetic='jacobi',
                     configs='single(6,6)',
                     use_jastrow=True,
                     include_all_mo=True)

        mo = torch.rand(10, 22, 45)
        det_explicit = wf.pool.det_explicit(mo)
        det_single = wf.pool.det_single(mo)
        assert(torch.allclose(det_explicit, det_single))


if __name__ == "__main__":
    # unittest.main()
    t = TestSlater()
    t.setUp()
    t.test_single()
    t.test_single_all_mo()
