from types import SimpleNamespace
import numpy as np 

class CalculatorBase(object):
    def __init__(self, atoms, atom_coords, basis, scf, units):

        self.atoms = atoms
        self.atom_coords = atom_coords
        self.basis_name = basis
        self.scf = scf 
        self.units = units
        self.out_file = None
        self.basis = self.init_basis()

    @staticmethod
    def init_basis():
        basis = SimpleNamespace()
        basis.spherical_harmonics_type = None
        basis.nshells = []
        basis.index_ctr = []
        basis.bas_exp, basis.bas_coeffs = [], []
        basis.bas_n, basis.bas_l, basis.bas_m = [], [], []
        return basis

    def run(self):
        raise NotImplementedError('Implement a run method in your calculator.')

    def get_basis(self):
        raise NotImplementedError('Implement a get_basis method in your calculator.')

    def get_mo_coeffs(self):
        raise NotImplementedError('Implement a get_mo_coeffs method in your calculator.')

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