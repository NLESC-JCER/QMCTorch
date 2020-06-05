import torch
import torch.optim as optim

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.solver import SolverOrbital
from qmctorch.sampler import Metropolis, Hamiltonian
from qmctorch.utils import plot_energy, plot_data, plot_walkers_traj, plot_block

import platform
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestRadialSlater(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)
        np.random.seed(0)

        self.mol = Molecule(atom='C 0 0 0; O 0 0 2.190; O 0 0 -2.190',
                            calculator='adf',
                            basis='dzp',
                            unit='bohr')

        # wave function
        self.wf = Orbital(self.mol, kinetic='jacobi',
                          configs='ground_state',
                          use_jastrow=True,
                          include_all_mo=False)

        npts = 1000
        self.pos = torch.zeros(npts, self.mol.nelec * 3)
        self.pos[:, 2] = torch.linspace(-4, 4, npts)
        self.dz = self.pos[1, 2] - self.pos[0, 2]

    def test_slater(self):

        xyz, r = self.wf.ao._process_position(self.pos)
        R, dR, d2R = self.wf.ao.radial(r, self.wf.ao.bas_n,
                                       self.wf.ao.bas_exp,
                                       xyz=xyz,
                                       derivative=[0, 1, 2],
                                       jacobian=False)

        R = R.detach().numpy()
        dR = dR.detach().numpy()
        d2R = d2R.detach().numpy()
        ielec = 0

        for iorb in range(7):

            r0 = R[:, ielec, iorb]

            dz_r0 = dR[:, ielec, iorb, 2]
            dz_r0_fd = np.gradient(r0, self.dz)

            delta = np.delete(np.abs(dz_r0-dz_r0_fd), np.s_[450:550])
            assert(np.all(delta < 1E-3))

            # plt.plot(r0)
            # plt.plot(dz_r0)
            # plt.plot(np.gradient(r0, self.dz))
            # plt.show()

        d2z_r0 = np.gradient(dR[:, 0, 0, 2], self.dz)
        plt.plot(d2z_r0)
        plt.plot(d2R[:, 0, 0])
        plt.show()

        return R, dR, d2R


if __name__ == "__main__":
    # unittest.main()

    t = TestRadialSlater()
    t.setUp()
    R, dR, d2R = t.test_slater()
