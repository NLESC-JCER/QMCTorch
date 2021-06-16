import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow

from ...path_utils import PATH_TEST

__PLOT__ = False


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


def generate_cube_files(t21file):

    nao = create_ao_variable_in_t21(t21file)
    create_densf_input_file(t21file, nao)
    os.system('$ADFBIN/densf < densf_input')


def create_ao_variable_in_t21(t21file):

    from scm import plams
    with plams.kFFile(t21file) as kf:
        nao = kf.read('Basis', 'naos')
        for iao in range(nao):

            var = [0.] * nao
            var[iao] = 1.
            name = 'AO%d' % iao
            kf.write('Basis', name, var)

    return nao


def create_densf_input_file(t21name, nao):

    f = open('densf_input', 'w')
    f.write('INPUTFILE %s\n\nCUBOUTPUT C_AO_\n\n' % t21name)

    f.write('GRID \n')
    f.write(' -1 -1 0\n')
    f.write(' 21 21\n')
    f.write(' 1 0 0 2\n')
    f.write(' 0 1 0 2\n')
    f.write('END\n\n')

    f.write('Orbitals GenBas\n')
    for orb_index in range(nao):
        f.write('  Basis%%AO%d\n' % orb_index)
    f.write('End\n\n')


class TestAOvaluesADF(unittest.TestCase):

    def setUp(self):

        # define the molecule
        path_hdf5 = (
            PATH_TEST / 'hdf5/C_adf_dzp.hdf5').absolute().as_posix()
        self.mol = Molecule(load=path_hdf5)

        # define the wave function
        self.wf = SlaterJastrow(self.mol, include_all_mo=True)

        # define the grid points
        self.npts = 21
        pts = get_pts(self.npts)

        self.pos = torch.zeros(self.npts**2, self.mol.nelec * 3)
        self.pos[:, :3] = pts
        self.pos = Variable(self.pos)
        self.pos.requires_grad = True

    def test_ao(self):

        aovals = self.wf.ao(self.pos).detach().numpy()

        for iorb in range(self.mol.basis.nao):

            path_cube = PATH_TEST / f'cube/C_AO_%Basis%AO{iorb}.cub'
            fname = path_cube.absolute().as_posix()
            adf_ref_data = np.array(read_cubefile(
                fname)).reshape(self.npts, self.npts)
            qmctorch_data = (aovals[:, 0, iorb]).reshape(
                self.npts, self.npts)

            delta = np.abs(adf_ref_data - qmctorch_data)

            if __PLOT__:
                plt.subplot(1, 3, 1)
                plt.imshow(adf_ref_data)

                plt.subplot(1, 3, 2)
                plt.imshow(qmctorch_data)

                plt.subplot(1, 3, 3)
                plt.imshow(delta)
                plt.show()

            assert(delta.mean() < 1E-3)


if __name__ == "__main__":
    unittest.main()
    # t = TestAOvaluesADF()
    # t.setUp()
    # t.test_ao()
    # t.test_mo()
