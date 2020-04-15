from types import SimpleNamespace
import h5py
import numpy as np


class CalculatorBase(object):
    def __init__(self, atoms, atom_coords, basis, scf, units, molname, calcname):

        self.atoms = atoms
        self.atom_coords = atom_coords
        self.basis_name = basis
        self.scf = scf
        self.units = units
        self.molname = molname
        self.calcname = calcname
        self.out_file = self.get_output_name()
        self.basis = SimpleNamespace()

    def run(self):
        raise NotImplementedError(
            'Implement a run method in your calculator.')

    def save_data(self):
        raise NotImplementedError(
            'Implement a save_data method in your calculator.')

    def check_h5file(self):
        """Check if the hdf5 contains all the necessary fields."""
        h5 = h5py.File(self.out_file, 'r')

        names = ['bas_coeffs', 'bas_exp', 'nshells',
                 'atom_coords_internal', 'nao', 'nmo',
                 'index_ctr', 'mos', 'TotalEnergy']

        if h5['harmonics_type'] == 'cart':
            names += ['bas_kx', 'bas_ky', 'bas_kz', 'bas_kr']

        elif h5['harmonics_type'] == 'sph':
            names += ['bas_n', 'bas_l', 'bas_m']

        for n in names:
            assert(n in h5.keys())

        h5.close()

    def get_basis(self):
        """Get the basis information needed to compute the AO values."""

        h5 = h5py.File(self.out_file, 'r')

        self.basis.radial_type = h5['radial_type'][()]
        self.basis.harmonics_type = h5['harmonics_type'][()]

        self.basis.nao = int(h5['nao'][()])
        self.basis.nmo = int(h5['nmo'][()])

        self.basis.nshells = h5['nshells'][()]
        self.basis.index_ctr = h5['index_ctr'][()]

        self.basis.bas_exp = h5['bas_exp'][()]
        self.basis.bas_coeffs = h5['bas_coeff'][()]

        self.basis.atom_coords_internal = h5['atom_coords_internal'][(
        )]

        if self.basis.harmonics_type == 'cart':
            self.basis.bas_kr = h5['bas_kr'][()]
            self.basis.bas_kx = h5['bas_kx'][()]
            self.basis.bas_ky = h5['bas_ky'][()]
            self.basis.bas_kz = h5['bas_kz'][()]

        elif self.basis.harmonics_type == 'sph':
            self.basis.bas_n = h5['bas_n'][()]
            self.basis.bas_l = h5['bas_l'][()]
            self.basis.bas_m = h5['bas_m'][()]

        h5.close()
        return self.basis

    def get_mo_coeffs(self):
        """Get the molecule orbital coefficients."""

        h5 = h5py.File(self.out_file, 'r')
        return h5['mos'][()]

    def print_total_energy(self):
        """Print the total energy."""
        h5 = h5py.File(self.out_file, 'r')
        e = h5['TotalEnergy'][()]
        print('== SCF Energy : ', e)
        h5.close()

    @staticmethod
    def normalize_columns(mat):
        """Normalize a matrix column-wise.

        Arguments:
            mat {np.ndarray} -- the matrix to be normalized

        Returns:
            np.ndarray -- normalized matrix
        """
        norm = np.sqrt((mat**2).sum(0))
        return mat / norm

    def get_output_name(self):
        return '_'.join([self.molname, self.calcname, self.basis_name]) + '.hdf5'
