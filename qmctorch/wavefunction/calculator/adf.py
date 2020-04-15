import numpy as np
import os
import shutil
import warnings
from .calculator_base import CalculatorBase
import h5py

try:
    from scm import plams
except ModuleNotFoundError:
    warnings.warn('scm python module not found')


class CalculatorADF(CalculatorBase):

    def __init__(self, atoms, atom_coords, basis, scf, units, molname):

        CalculatorBase.__init__(
            self, atoms, atom_coords, basis, scf, units, molname, 'adf')
        self.run()

    def run(self):
        """Run the calculation using ADF."""

        if os.path.isfile(self.out_file):
            print('Reusing previous calculation from ', self.out_file)

        else:

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
            self.save_data(t21_path)
            shutil.rmtree(plams_wd)

        # print energy
        self.print_total_energy()

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

    def save_data(self, kffile):
        """Save the basis information needed to compute the AO values."""

        kf = plams.KFFile(kffile)
        h5 = h5py.File(self.out_file, 'w')

        h5['TotalEnergy'] = kf.read('Total Energy', 'Total energy')
        h5['radial_type'] = 'sto'
        h5['harmonics_type'] = 'cart'

        nao = kf.read('Basis', 'naos')
        nmo = kf.read('A', 'nmo_A')
        h5['nao'] = nao
        h5['nmo'] = nmo

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

        h5.create_dataset('nshells', data=basis_nshells)
        h5.create_dataset('index_ctr', data=np.arange(nao))

        h5.create_dataset('bas_kx', data=basis_bas_kx)
        h5.create_dataset('bas_ky', data=basis_bas_ky)
        h5.create_dataset('bas_kz', data=basis_bas_kz)
        h5.create_dataset('bas_kr', data=basis_bas_kr)

        h5.create_dataset('bas_exp', data=basis_bas_exp)
        h5.create_dataset(
            'bas_coeff', data=np.ones_like(basis_bas_exp))
        h5.create_dataset('bas_norm', data=basis_bas_norm)

        h5.create_dataset('atom_coords_internal', data=np.array(
            kf.read('Geometry', 'xyz')).reshape(-1, 3))

        # Molecular orbitals
        mos = np.array(kf.read('A', 'Eigen-Bas_A'))
        mos = mos.reshape(nmo, nao).T
        mos = self.normalize_columns(mos)
        h5.create_dataset('mos', data=mos)

        # close and check
        h5.close()
        self.check_h5file(self.out_file)
