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

    def run(self):
        raise NotImplementedError(
            'Implement a run method in your calculator.')

    def save_data(self):
        raise NotImplementedError(
            'Implement a save_data method in your calculator.')

    def check_h5file(self):
        """Check if the hdf5 contains all the necessary fields."""
        h5 = h5py.File(self.hdf5file, 'r')
        calc_grp = h5['calculator']

        names = ['bas_coeff', 'bas_exp', 'nshells',
                 'atom_coords_internal', 'nao', 'nmo',
                 'index_ctr', 'mos', 'TotalEnergy']

        if calc_grp['harmonics_type'] == 'cart':
            names += ['bas_kx', 'bas_ky', 'bas_kz', 'bas_kr']

        elif calc_grp['harmonics_type'] == 'sph':
            names += ['bas_n', 'bas_l', 'bas_m']

        for n in names:
            if n not in calc_grp.keys():
                raise KeyError(n, ' not in the hdf5 file')

        h5.close()

    def get_basis(self):
        """Get the basis information needed to compute the AO values."""

        h5 = h5py.File(self.hdf5file, 'r')
        basis_grp = h5['basis']

        self.basis.radial_type = basis_grp['radial_type'][()]
        self.basis.harmonics_type = basis_grp['harmonics_type'][()]

        self.basis.nao = int(basis_grp['nao'][()])
        self.basis.nmo = int(basis_grp['nmo'][()])

        self.basis.nshells = basis_grp['nshells'][()]
        self.basis.index_ctr = basis_grp['index_ctr'][()]

        self.basis.bas_exp = basis_grp['bas_exp'][()]
        self.basis.bas_coeffs = basis_grp['bas_coeff'][()]

        self.basis.atom_coords_internal = basis_grp['atom_coords_internal'][(
        )]

        if self.basis.harmonics_type == 'cart':
            self.basis.bas_kr = basis_grp['bas_kr'][()]
            self.basis.bas_kx = basis_grp['bas_kx'][()]
            self.basis.bas_ky = basis_grp['bas_ky'][()]
            self.basis.bas_kz = basis_grp['bas_kz'][()]

        elif self.basis.harmonics_type == 'sph':
            self.basis.bas_n = basis_grp['bas_n'][()]
            self.basis.bas_l = basis_grp['bas_l'][()]
            self.basis.bas_m = basis_grp['bas_m'][()]

        h5.close()
        return self.basis

    def get_mo_coeffs(self):
        """Get the molecule orbital coefficients."""

        h5 = h5py.File(self.hdf5file, 'r')
        return h5['calculator']['mos'][()]

    def print_total_energy(self):
        """Print the total energy."""
        h5 = h5py.File(self.hdf5file, 'r')
        e = h5['calculator']['TotalEnergy'][()]
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
