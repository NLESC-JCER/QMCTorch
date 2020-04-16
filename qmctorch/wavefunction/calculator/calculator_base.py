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
