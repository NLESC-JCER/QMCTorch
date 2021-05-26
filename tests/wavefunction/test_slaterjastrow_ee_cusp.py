from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow
from qmctorch.utils import set_torch_double_precision
from qmctorch.wavefunction.jastrows.elec_elec.kernels import FullyConnectedJastrowKernel

import numpy as np
import torch
import unittest

torch.set_default_tensor_type(torch.DoubleTensor)


class TestSlaterJastrowElectronCusp(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        mol = Molecule(
            atom='He 0.5 0 0; He -0.5 0 0',
            unit='bohr',
            calculator='pyscf',
            basis='sto-3g',
            redo_scf=True)

        self.wf = SlaterJastrow(mol,
                                jastrow_kernel=FullyConnectedJastrowKernel,
                                kinetic='jacobi',
                                include_all_mo=True,
                                configs='ground_state')

        self.nbatch = 100

    def test_ee_cusp(self):

        import matplotlib.pyplot as plt
        pos_x = torch.Tensor(np.random.rand(
            self.nbatch,  self.wf.nelec, 3))
        x = torch.linspace(0, 2, self.nbatch)
        pos_x[:, 0, :] = torch.as_tensor([0., 0., 0.]) + 1E-6
        pos_x[:, 1, 0] = 0.
        pos_x[:, 1, 1] = 0.
        pos_x[:, 1, 2] = x

        pos_x[:, 2, :] = 0.5*torch.as_tensor([1., 1., 1.])
        pos_x[:, 3, :] = -0.5*torch.as_tensor([1., 1., 1.])

        pos_x = pos_x.reshape(self.nbatch, self.wf.nelec*3)
        pos_x.requires_grad = True

        x = x.detach().numpy()
        j = self.wf.jastrow(pos_x).detach().numpy()
        plt.plot(x, j)
        plt.show()

        dx = x[1]-x[0]
        dj = (j[1:]-j[0:-1])/dx

        plt.plot(x[:-1], dj/j[:-1])
        plt.show()

        epot = self.wf.electronic_potential(pos_x).detach().numpy()
        ekin = self.wf.kinetic_energy_jacobi(pos_x).detach().numpy()
        eloc = self.wf.local_energy(pos_x).detach().numpy()
        plt.plot(x, epot)
        plt.plot(x, ekin)
        plt.plot(x, eloc)
        plt.show()


if __name__ == "__main__":
    # unittest.main()
    t = TestSlaterJastrowElectronCusp()
    t.setUp()
    t.test_ee_cusp()
