import torch
from torch.autograd import Variable
from qmctorch.wavefunction import Orbital, Molecule
from pyscf import gto

import numpy as np
import unittest

import matplotlib.pyplot as plt

import os


class TestAOvaluesPyscf(unittest.TestCase):

    def setUp(self):

        # define the molecule
        at = 'C 0 0 0'
        basis = 'dzp'
        self.mol = Molecule(atom=at,
                            calculator='pyscf',
                            basis=basis,
                            unit='bohr')

        self.m = gto.M(atom=at, basis=basis, unit='bohr')

        # define the wave function
        self.wf = Orbital(self.mol)

        self.pos = torch.zeros(100, self.mol.nelec * 3)

        self.pos[:, 0] = torch.linspace(-5, 5, 100)
        self.pos[:, 1] = torch.linspace(-5, 5, 100)
        self.pos[:, 2] = torch.linspace(-5, 5, 100)

        self.pos = Variable(self.pos)
        self.pos.requires_grad = True

        self.x = self.pos[:, 0].detach().numpy()

    def test_ao(self):

        nzlm = np.linalg.norm(self.m.cart2sph_coeff(), axis=1)

        aovals = self.wf.ao(self.pos).detach().numpy()/nzlm
        aovals_ref = self.m.eval_ao('GTOval_cart',
                                    self.pos.detach().numpy()[:, :3])

        for iorb in range(self.mol.basis.nao):

            plt.plot(self.x, aovals[:, 0, iorb])
            plt.plot(self.x, aovals_ref[:, iorb])
            plt.show()

            assert np.allclose(
                aovals[:, 0, iorb], aovals_ref[:, iorb])

    def test_ao_deriv(self):

        nzlm = np.linalg.norm(self.m.cart2sph_coeff(), axis=1)

        daovals = self.wf.ao(
            self.pos, derivative=1).detach().numpy()/nzlm

        daovals_ref = self.m.eval_gto(
            'GTOval_ip_cart', self.pos.detach().numpy()[:, :3])
        daovals_ref = daovals_ref.sum(0)

        for iorb in range(self.mol.basis.nao):

            plt.plot(self.x, daovals[:, 0, iorb])
            plt.plot(self.x, daovals_ref[:, iorb])
            plt.show()

            assert np.allclose(
                daovals[:, 0, iorb], daovals_ref[:, iorb])


if __name__ == "__main__":
    unittest.main()
