import torch
import torch.optim as optim

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.utils import InterpolateAtomicOribtals, InterpolateMolecularOrbitals
import platform

import numpy as np
import unittest


class TestInterpolate(unittest.TestCase):

    def setUp(self):

        # molecule
        self.mol = Molecule(
            atom='H 0 0 -0.69; H 0 0 0.69',
            unit='bohr',
            calculator='pyscf',
            basis='dzp')

        # wave function
        self.wf = Orbital(self.mol, kinetic='jacobi',
                          configs='single(2,2)',
                          use_jastrow=True)

        npts = 51
        self.pos = torch.zeros(npts, 6)
        self.pos[:, 2] = torch.linspace(-2, 2, npts)

    def test_ao(self):

        interp_ao = InterpolateAtomicOribtals(self.wf)
        inter = interp_ao(self.pos)
        ref = self.wf.ao(self.pos)
        delta = (inter-ref).abs().mean()
        assert(delta < 0.1)

    def test_mo_reg(self):

        interp_mo = InterpolateMolecularOrbitals(self.wf)
        inter = interp_mo(self.pos, method='reg')
        ref = self.wf.mo(self.wf.mo_scf(self.wf.ao(self.pos)))
        delta = (inter-ref).abs().mean()
        assert(delta < 0.1)

    def test_mo_irreg(self):

        interp_mo = InterpolateMolecularOrbitals(self.wf)
        inter = interp_mo(self.pos, method='irreg')
        ref = self.wf.mo(self.wf.mo_scf(self.wf.ao(self.pos)))
        delta = (inter-ref).abs().mean()
        assert(delta < 0.1)


if __name__ == "__main__":
    # unittest.main()
    t = TestInterpolate()
    t.setUp()
    t.test_ao()
    t.test_mo_reg()
    t.test_mo_irreg()
