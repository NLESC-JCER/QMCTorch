import torch
from torch.autograd import Variable
from qmctorch.wavefunction import Orbital, Molecule
from pyscf import gto

import numpy as np
import unittest

import matplotlib.pyplot as plt

import os


def read_cubefile(fname):
    with open(fname, 'r') as f:
        data = f.readlines()
    vals = []
    for d in data[7:]:
        vals.append(float(d.split('\n')[0]))
    return vals


def get_pts(npts):
    x = torch.linspace(-1, 1, npts)
    xx, yy = torch.meshgrid((x, x))
    pts = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)
    pts = torch.cat((pts, torch.zeros(pts.shape[0], 1)), axis=1)
    return pts


def generate_cube_files(t21file, npts):
    create_densf_input_file(t21file, npts)
    os.system('$ADFBIN/densf < densf_input')


def create_densf_input_file(t21name, npts):

    f = open('densf_input', 'w')
    f.write('INPUTFILE %s\n\nCUBOUTPUT MO_\n\n' % t21name)

    f.write('GRID \n')
    f.write(' -1 -1 0\n')
    f.write(' %d %d\n' % (npts, npts))
    f.write(' 1 0 0 2\n')
    f.write(' 0 1 0 2\n')
    f.write('END\n\n')

    f.write('Orbitals SCF\n')
    f.write('    A occ\n')
    f.write('    A virt\n')
    f.write('End\n\n')


class TestMOvaluesADF(unittest.TestCase):

    def setUp(self):

        # define the molecule
        self.mol = Molecule(load='hdf5/C_adf_dzp.hdf5')

        # define the wave function
        self.wf = Orbital(self.mol, include_all_mo=True)

        # define the grid points
        self.npts = 21
        pts = get_pts(self.npts)

        self.pos = torch.zeros(self.npts**2, self.mol.nelec * 3)
        self.pos[:, :3] = pts
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True

    def test_mo(self):

        movals = self.wf.mo_scf(self.wf.ao(self.pos)).detach().numpy()

        for iorb in range(self.mol.basis.nmo):

            fname = 'cube/C_MO_%%SCF_A%%%d.cub' % (iorb+1)
            adf_ref_data = np.array(read_cubefile(
                fname)).reshape(self.npts, self.npts)**2
            qmctorch_data = (movals[:, 0, iorb]).reshape(
                self.npts, self.npts)**2

            delta = np.abs(adf_ref_data - qmctorch_data)

            plt.subplot(1, 3, 1)
            plt.imshow(adf_ref_data)

            plt.subplot(1, 3, 2)
            plt.imshow(qmctorch_data)

            plt.subplot(1, 3, 3)
            plt.imshow(delta)
            plt.show()

            # assert(delta.mean() < 1E-3)

    # def test_ao_deriv(self):

    #     ip_aovals = self.wf.ao(
    #         self.pos, derivative=1).detach().numpy()
    #     ip_aovals_ref = self.m.eval_gto(
    #         'GTOval_ip_cart', self.pos.detach().numpy()[:, :3])
    #     ip_aovals_ref = ip_aovals_ref.sum(0)

    #     assert np.allclose(ip_aovals[:, 0, self.iorb],
    #                        ip_aovals_ref[:, self.iorb])

    # def test_ao_hess(self):

    #     i2p_aovals = self.wf.ao(
    #         self.pos, derivative=2).detach().numpy()

    #     ip_aovals = self.wf.ao(
    #         self.pos, derivative=1).detach().numpy()

    #     path = os.path.dirname(os.path.realpath(__file__))
    #     i2p_aovals_ref = np.loadtxt(path + '/hess_ao_h2.dat')

    #     assert np.allclose(
    #         i2p_aovals[:, 0, self.iorb], i2p_aovals_ref)

    # def test_all_der(self):
    #     aovals = self.wf.ao(self.pos, derivative=[
    #                         0, 1, 2])


if __name__ == "__main__":
    # unittest.main()
    t = TestMOvaluesADF()
    t.setUp()
    t.test_mo()
