import numpy as np
import os
import shutil
import warnings
from .calculator_base import CalculatorBase
import h5py
from types import SimpleNamespace

try:
    from scm import plams
except ModuleNotFoundError:
    warnings.warn('scm python module not found')


class CalculatorADF(CalculatorBase):

    def __init__(self, atoms, atom_coords, basis, scf, units, molname):

        CalculatorBase.__init__(
            self, atoms, atom_coords, basis, scf, units, molname, 'adf')

    def run(self):
        """Run the calculation using ADF."""

        # path needed for the calculation
        wd = ''.join(self.atoms) + '_' + self.basis_name
        t21_name = wd + '.t21'
        plams_wd = './plams_workdir'
        t21_path = os.path.join(
            plams_wd, os.path.join(wd, t21_name))

        # configure plams and run the calculation
        self.init_plams()
        mol = self.get_plams_molecule()
        sett = self.get_plams_settings()
        job = plams.ADFJob(molecule=mol, settings=sett, name=wd)
        job.run()

        # extract the data to hdf5
        basis = self.get_basis_data(t21_path)
        shutil.rmtree(plams_wd)

        return basis

    def init_plams(self):
        """Init PLAMS."""
        plams.init()
        plams.config.log.stdout = -1
        plams.config.erase_workdir = True

    def get_plams_molecule(self):
        """Returns a plams molecule object."""
        mol = plams.Molecule()
        for at, xyz in zip(self.atoms, self.atom_coords):
            mol.add_atom(plams.Atom(symbol=at, coords=tuple(xyz)))
        return mol

    def get_plams_settings(self):
        """Returns a plams setting object."""

        sett = plams.Settings()
        sett.input.basis.type = self.basis_name.upper()
        sett.input.basis.core = 'None'
        sett.input.symmetry = 'nosym'
        sett.input.XC.HartreeFock = ''

        # correct unit
        if self.units == 'angs':
            sett.input.units.length = 'Angstrom'
        elif self.units == 'bohr':
            sett.input.units.length = 'Bohr'

        # total energy
        sett.input.totalenergy = True

        return sett

    def get_basis_data(self, kffile):
        """Save the basis information needed to compute the AO values."""

        kf = plams.KFFile(kffile)

        basis = SimpleNamespace()

        basis.TotalEnergy = kf.read('Total Energy', 'Total energy')
        basis.radial_type = 'sto'
        basis.harmonics_type = 'cart'

        nao = kf.read('Basis', 'naos')
        nmo = kf.read('A', 'nmo_A')
        basis.nao = nao
        basis.nmo = nmo

        # number of bas per atom type
        nbptr = kf.read('Basis', 'nbptr')

        # number of atom per atom typ
        nqptr = kf.read('Geometry', 'nqptr')
        atom_type = kf.read('Geometry', 'atomtype').split()

        # number of bas per atom type
        nshells = np.array([nbptr[i] - nbptr[i - 1]
                            for i in range(1, len(nbptr))])

        # kx/ky/kz/kr exponent per atom type
        bas_kx = np.array(kf.read('Basis', 'kx'))
        bas_ky = np.array(kf.read('Basis', 'ky'))
        bas_kz = np.array(kf.read('Basis', 'kz'))
        bas_kr = np.array(kf.read('Basis', 'kr'))

        # bas exp/coeff/norm per atom type
        bas_exp = np.array(kf.read('Basis', 'alf'))
        bas_norm = np.array(kf.read('Basis', 'bnorm'))

        basis_nshells = []
        basis_bas_kx, basis_bas_ky, basis_bas_kz = [], [], []
        basis_bas_kr = []
        basis_bas_exp, basis_bas_norm = [], []

        for iat, at in enumerate(atom_type):

            number_copy = nqptr[iat + 1] - nqptr[iat]
            idx_bos = list(range(nbptr[iat] - 1, nbptr[iat + 1] - 1))

            basis_nshells += [nshells[iat]] * number_copy

            basis_bas_kx += list(bas_kx[idx_bos]) * number_copy
            basis_bas_ky += list(bas_ky[idx_bos]) * number_copy
            basis_bas_kz += list(bas_kz[idx_bos]) * number_copy
            basis_bas_kr += list(bas_kr[idx_bos]) * number_copy
            basis_bas_exp += list(bas_exp[idx_bos]) * number_copy
            basis_bas_norm += list(
                bas_norm[idx_bos]) * number_copy

        basis.nshells = basis_nshells
        basis.index_ctr = np.arange(nao)

        basis.bas_kx = np.array(basis_bas_kx)
        basis.bas_ky = np.array(basis_bas_ky)
        basis.bas_kz = np.array(basis_bas_kz)
        basis.bas_kr = np.array(basis_bas_kr)

        basis.bas_exp = np.array(basis_bas_exp)
        basis.bas_coeffs = np.ones_like(basis_bas_exp)
        basis.bas_norm = np.array(basis_bas_norm)

        basis.atom_coords_internal = np.array(
            kf.read('Geometry', 'xyz')).reshape(-1, 3)

        # Molecular orbitals
        mos = np.array(kf.read('A', 'Eigen-Bas_A'))
        mos = mos.reshape(nmo, nao).T
        mos = self.normalize_columns(mos)
        basis.mos = mos

        return basis
