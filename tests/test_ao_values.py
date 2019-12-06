import torch
from torch.autograd import Variable
from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.wavefunction.molecule import Molecule
from pyscf import gto

import numpy as np
import unittest


class TestAOvalues(unittest.TestCase):

    def setUp(self):

        # define the molecule
        at = 'H 0 0 0; H 0 0 1'
        self.mol = Molecule(atom=at,
                            basis_type='gto',
                            basis='sto-3g',
                            unit='bohr')

        self.m = gto.M(atom=at, basis='sto-3g', unit='bohr')

        # define the wave function
        self.wf = Orbital(self.mol)

        self.pos = torch.zeros(100, self.mol.nelec*3)
        self.pos[:, 2] = torch.linspace(-5, 5, 100)

        self.pos = Variable(self.pos)
        self.pos.requires_grad = True
        self.iorb = 0
        self.x = self.pos[:, 2].detach().numpy()

    def test_ao(self):

        aovals = self.wf.ao(self.pos).detach().numpy()
        aovals_ref = self.m.eval_gto(
            'GTOval_cart', self.pos.detach().numpy()[:, :3])

        assert np.allclose(aovals[:, 0, self.iorb], aovals_ref[:, self.iorb])

    def test_ao_deriv(self):

        ip_aovals = self.wf.ao(self.pos, derivative=1).detach().numpy()
        ip_aovals_ref = self.m.eval_gto(
            'GTOval_ip_cart', self.pos.detach().numpy()[:, :3])
        ip_aovals_ref = ip_aovals_ref.sum(0)

        assert np.allclose(ip_aovals[:, 0, self.iorb],
                           ip_aovals_ref[:, self.iorb])

    def test_ao_hess(self):

        i2p_aovals = self.wf.ao(self.pos, derivative=2).detach().numpy()

        ip_aovals_ref = self.m.eval_gto(
            'GTOval_ip_cart', self.pos.detach().numpy()[:, :3])
        ip_aovals_ref = ip_aovals_ref.sum(0)

        i2p_aovals_ref = np.gradient(ip_aovals_ref[:, self.iorb], self.x)

        # assert np.allclose(i2p_aovals[:,0,self.iorb],i2p_aovals_ref)


if __name__ == "__main__":
    unittest.main()
