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
