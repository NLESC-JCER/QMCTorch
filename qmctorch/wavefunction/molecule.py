import os
import math
import numpy as np
from mendeleev import element
from types import SimpleNamespace
import h5py

from .calculator.adf import CalculatorADF
from .calculator.pyscf import CalculatorPySCF

from ..utils import dump_to_hdf5


class Molecule(object):

    def __init__(self, atom, calculator='adf',
                 scf='hf', basis='dzp', unit='bohr',
                 name=None):

        self.atoms_str = atom
        self.unit = unit
        self.max_angular = 2
        self.atoms = []
        self.atom_coords = []
        self.atomic_number = []
        self.atomic_nelec = []
        self.nelec = 0
        self.name = name

        if self.unit not in ['angs', 'bohr']:
            raise ValueError('unit should be angs or bohr')

        self.process_atom_str()

        self.hdf5file = '_'.join(
            [self.name, calculator, basis]) + '.hdf5'

        if os.path.isfile(self.hdf5file):
            print('Reusing data from ', self.hdf5file)
            self.basis = self.load_basis()

        else:
            calc = {'adf': CalculatorADF,
                    'pyscf': CalculatorPySCF}[calculator]

            self.calculator = calc(
                self.atoms, self.atom_coords, basis, scf, self.unit, self.name)

            self.basis = self.calculator.run()

            dump_to_hdf5(self, self.hdf5file, root_name='molecule')

        self.check_basis()

    def process_atom_str(self):
        '''Process the input file.'''

        if os.path.isfile(self.atoms_str):
            atoms = self.read_xyz_file()
        else:
            atoms = self.atoms_str.split(';')

        self.get_atomic_properties(atoms)

    def get_atomic_properties(self, atoms):
        """Get the properties of all atoms in the molecule

        Arguments:
            atoms {list} -- atoms and xyz position
        """

        # loop over all atoms
        for a in atoms:
            atom_data = a.split()
            self.atoms.append(atom_data[0])
            x, y, z = float(atom_data[1]), float(
                atom_data[2]), float(atom_data[3])

            conv2bohr = 1
            if self.unit == 'angs':
                conv2bohr = 1.88973
            self.atom_coords.append(
                [x * conv2bohr, y * conv2bohr, z * conv2bohr])

            self.atomic_number.append(
                element(atom_data[0]).atomic_number)
            self.atomic_nelec.append(element(atom_data[0]).electrons)
            self.nelec += element(atom_data[0]).electrons

        # size of the system
        self.natom = len(self.atoms)
        if self.nelec % 2 != 0:
            raise ValueError("Only equal spin up/down supported.")
        self.nup = math.ceil(self.nelec / 2)
        self.ndown = math.floor(self.nelec / 2)

        # name of the system
        if self.name is None:
            self.name = self.get_mol_name(self.atoms)

    def read_xyz_file(self):
        """Process a xyz file containing the data

        Returns:
            list -- atoms and xyz position
        """
        with open(self.atoms_str, 'r') as f:
            data = f.readlines()
        atoms = data[2:]
        self.atoms_str = ''
        for a in atoms[:-1]:
            self.atoms_str += a + '; '
        self.atoms_str += atoms[-1]
        return atoms

    def domain(self, method):
        """Define the walker initialization method

        Arguments:
            method str -- 'center'  : all electron at the center of the system
                          'uniform' : all electrons in a box
                                      covering the system
                          'normal   : all electrons in a shpere
                                      covering the system
                          'atomic'  : electrons around atoms

        Raises:
            ValueError: if method is not supported

        Returns:
            dict -- data required to initialize the walkers
        """
        domain = dict()

        if method == 'center':
            domain['center'] = np.mean(self.atom_coords, 0)

        elif method == 'uniform':
            domain['min'] = np.min(self.atom_coords) - 0.5
            domain['max'] = np.max(self.atom_coords) + 0.5

        elif method == 'normal':
            domain['mean'] = np.mean(self.atom_coords, 0)
            domain['sigma'] = np.diag(
                np.std(self.atom_coords, 0) + 0.25)

        elif method == 'atomic':
            domain['atom_coords'] = self.atom_coords
            domain['atom_num'] = self.atomic_number
            domain['atom_nelec'] = self.atomic_nelec

        else:
            raise ValueError(
                'Method to initialize the walkers not recognized')

        return domain

    @staticmethod
    def get_mol_name(atoms):
        mol_name = ''
        unique_atoms = list(set(atoms))
        for ua in unique_atoms:
            mol_name += ua + str(atoms.count(ua))
        return mol_name

    def load_basis(self):
        """Get the basis information needed to compute the AO values."""

        h5 = h5py.File(self.hdf5file, 'r')
        basis_grp = h5['molecule']['basis']
        self.basis = SimpleNamespace()

        self.basis.radial_type = basis_grp['radial_type'][()]
        self.basis.harmonics_type = basis_grp['harmonics_type'][()]

        self.basis.nao = int(basis_grp['nao'][()])
        self.basis.nmo = int(basis_grp['nmo'][()])

        self.basis.nshells = basis_grp['nshells'][()]
        self.basis.index_ctr = basis_grp['index_ctr'][()]

        self.basis.bas_exp = basis_grp['bas_exp'][()]
        self.basis.bas_coeffs = basis_grp['bas_coeffs'][()]

        self.basis.atom_coords_internal = basis_grp['atom_coords_internal'][(
        )]

        self.basis.TotalEnergy = basis_grp['TotalEnergy'][()]
        self.basis.mos = basis_grp['mos'][()]

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

    def load_mo_coeffs(self):
        """Get the molecule orbital coefficients."""

        h5 = h5py.File(self.hdf5file, 'r')
        return h5['molecule']['basis']['mos'][()]

    def print_total_energy(self):
        """Print the total energy."""
        h5 = h5py.File(self.hdf5file, 'r')
        e = h5['molecule']['basis']['TotalEnergy'][()]
        print('== SCF Energy : ', e)
        h5.close()

    def check_basis(self):
        """Check if the basis contains all the necessary fields."""

        names = ['bas_coeffs', 'bas_exp', 'nshells',
                 'atom_coords_internal', 'nao', 'nmo',
                 'index_ctr', 'mos', 'TotalEnergy']

        if self.basis.harmonics_type == 'cart':
            names += ['bas_kx', 'bas_ky', 'bas_kz', 'bas_kr']

        elif self.basis.harmonics_type == 'sph':
            names += ['bas_n', 'bas_l', 'bas_m']

        for n in names:
            if not hasattr(self.basis, n):
                raise ValueError(n, ' not in the basis namespace')


if __name__ == "__main__":

    mol = Molecule(
        atom='H 0 0 -3.015; O 0 0 0; H 0 0 3.015',
        calculator='adf',
        basis='dzp',
        unit='bohr')
