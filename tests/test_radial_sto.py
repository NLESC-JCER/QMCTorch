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


def second_derivative(xm1, x0, xp1, eps):
    return (xm1 - 2*x0 + xp1) / eps/eps


class TestRadialSlater(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(0)
        np.random.seed(0)

        self.mol = Molecule(load='hdf5/CO2_adf_dzp.hdf5')

        # wave function
        self.wf = Orbital(self.mol, kinetic='jacobi',
                          configs='ground_state',
                          use_jastrow=True,
                          include_all_mo=False)

    def test_first_derivative_x(self):

        npts = 1000
        self.pos = torch.zeros(npts, self.mol.nelec * 3)
        self.pos[:, 0] = torch.linspace(-4, 4, npts)
        self.dx = self.pos[1, 0] - self.pos[0, 0]

        xyz, r = self.wf.ao._process_position(self.pos)
        R, dR = self.wf.ao.radial(r, self.wf.ao.bas_n,
                                  self.wf.ao.bas_exp,
                                  xyz=xyz,
                                  derivative=[0, 1],
                                  jacobian=False)

        R = R.detach().numpy()
        dR = dR.detach().numpy()
        ielec = 0

        for iorb in range(7):
            r0 = R[:, ielec, iorb]
            dz_r0 = dR[:, ielec, iorb, 0]
            dz_r0_fd = np.gradient(r0, self.dx)
            delta = np.delete(np.abs(dz_r0-dz_r0_fd), np.s_[450:550])

            # plt.plot(dz_r0)
            # plt.plot(dz_r0_fd)
            # plt.show()

            assert(np.all(delta < 1E-3))

    def test_first_derivative_y(self):

        npts = 1000
        self.pos = torch.zeros(npts, self.mol.nelec * 3)
        self.pos[:, 1] = torch.linspace(-4, 4, npts)
        self.dy = self.pos[1, 1] - self.pos[0, 1]

        xyz, r = self.wf.ao._process_position(self.pos)
        R, dR = self.wf.ao.radial(r, self.wf.ao.bas_n,
                                  self.wf.ao.bas_exp,
                                  xyz=xyz,
                                  derivative=[0, 1],
                                  jacobian=False)

        R = R.detach().numpy()
        dR = dR.detach().numpy()
        ielec = 0

        for iorb in range(7):
            r0 = R[:, ielec, iorb]
            dz_r0 = dR[:, ielec, iorb, 1]
            dz_r0_fd = np.gradient(r0, self.dy)
            delta = np.delete(np.abs(dz_r0-dz_r0_fd), np.s_[450:550])

            # plt.plot(dz_r0)
            # plt.plot(dz_r0_fd)
            # plt.show()

            assert(np.all(delta < 1E-3))

    def test_first_derivative_z(self):

        npts = 1000
        self.pos = torch.zeros(npts, self.mol.nelec * 3)
        self.pos[:, 2] = torch.linspace(-4, 4, npts)
        self.dz = self.pos[1, 2] - self.pos[0, 2]

        xyz, r = self.wf.ao._process_position(self.pos)
        R, dR = self.wf.ao.radial(r, self.wf.ao.bas_n,
                                  self.wf.ao.bas_exp,
                                  xyz=xyz,
                                  derivative=[0, 1],
                                  jacobian=False)
        R = R.detach().numpy()
        dR = dR.detach().numpy()
        ielec = 0

        for iorb in range(7):

            r0 = R[:, ielec, iorb]
            dz_r0 = dR[:, ielec, iorb, 2]
            dz_r0_fd = np.gradient(r0, self.dz)
            delta = np.delete(np.abs(dz_r0-dz_r0_fd), np.s_[450:550])

            # plt.plot(dz_r0)
            # plt.plot(dz_r0_fd)
            # plt.show()

            assert(np.all(delta < 1E-3))

    def test_laplacian(self, eps=1E-4):

        npts = 1000

        self.pos = torch.zeros(npts, self.mol.nelec * 3)
        self.pos[:, 2] = torch.linspace(-4, 4, npts)
        eps = self.pos[1, 2] - self.pos[0, 2]

        self.pos[:, 2] = torch.linspace(-4, 4, npts)

        self.pos[:, 3] = eps
        self.pos[:, 5] = torch.linspace(-4, 4, npts)

        self.pos[:, 6] = -eps
        self.pos[:, 8] = torch.linspace(-4, 4, npts)

        self.pos[:, 10] = eps
        self.pos[:, 11] = torch.linspace(-4, 4, npts)

        self.pos[:, 13] = -eps
        self.pos[:, 14] = torch.linspace(-4, 4, npts)

        xyz, r = self.wf.ao._process_position(self.pos)
        R, dR, d2R = self.wf.ao.radial(r, self.wf.ao.bas_n,
                                       self.wf.ao.bas_exp,
                                       xyz=xyz,
                                       derivative=[0, 1, 2],
                                       jacobian=False)

        for iorb in range(7):

            lap_analytic = np.zeros(npts-2)
            lap_fd = np.zeros(npts-2)

            for i in range(1, npts-1):
                lap_analytic[i-1] = d2R[i, 0, iorb]

                r0 = R[i, 0, iorb].detach().numpy()
                rpz = R[i+1, 0, iorb].detach().numpy()
                rmz = R[i-1, 0, iorb].detach().numpy()
                d2z = second_derivative(rmz, r0, rpz, eps)

                r0 = R[i, 0, iorb]
                rpx = R[i, 1, iorb]
                rmx = R[i, 2, iorb]
                d2x = second_derivative(rmx, r0, rpx, eps)

                r0 = R[i, 0, iorb]
                rpy = R[i, 3, iorb]
                rmy = R[i, 4, iorb]
                d2y = second_derivative(rmy, r0, rpy, eps)

                lap_fd[i-1] = d2x + d2y + d2z

            delta = np.delete(
                np.abs(lap_analytic-lap_fd), np.s_[450:550])

            assert(np.all(delta < 5E-3))

            # plt.plot(lap_analytic, linewidth=2)
            # plt.plot(lap_fd)
            # plt.show()


if __name__ == "__main__":
    unittest.main()

    # t = TestRadialSlater()
    # t.setUp()
    # t.test_first_derivative_x()
    # # t.test_first_derivative_y()
    # # t.test_first_derivative_z()
    # t.test_laplacian()
