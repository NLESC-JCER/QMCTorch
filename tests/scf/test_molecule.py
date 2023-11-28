from qmctorch.scf import Molecule
import numpy as np
import unittest


class TestMolecule(unittest.TestCase):

    def test1_create(self):

        # molecule
        mol = Molecule(
            atom='H 0. 0. 0.; H 0. 0. 1.',
            unit='bohr',
            scf='hf',
            calculator='pyscf',
            basis='sto-3g',
            redo_scf=True)

        mol.print_total_energy()

    def test2_load(self):
        mol = Molecule(load='H2_pyscf_sto-3g.hdf5')

    def test3_domain(self):
        mol = Molecule(load='H2_pyscf_sto-3g.hdf5')

        domain_center = mol.domain('center')
        assert (domain_center['center'] ==
                np.array([0., 0., 0.5])).all()

        domain_uniform = mol.domain('uniform')
        assert domain_uniform == {
            'method': 'uniform', 'min': -0.5, 'max': 1.5}

        domain_normal = mol.domain('normal')
        assert np.all(domain_normal['mean']
                      == np.array([0., 0., 0.5]))

        domain_atomic = mol.domain('atomic')
        assert np.all(domain_atomic['atom_coords'] == np.array([[0., 0., 0.],
                                                                [0., 0., 1.]]))

    def test4_create_dft(self):

        # molecule
        mol = Molecule(
            atom='H 0. 0. 0.; H 0. 0. 1.',
            unit='bohr',
            calculator='pyscf',
            scf='dft',
            basis='sto-3g',
            redo_scf=True)

        mol.print_total_energy()


if __name__ == "__main__":
    unittest.main()
