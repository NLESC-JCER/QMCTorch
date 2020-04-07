import os
import math
import numpy as np
from mendeleev import element


from deepqmc.wavefunction.calculator.adf import CalculatorADF

class Molecule(object):

    def __init__(self, atom, calculator=CalculatorADF, 
                 scf='hf', basis='dzp', unit='bohr'):

        self.atoms_str = atom
        self.unit = unit 
        self.max_angular = 2
        
        self.atoms = []
        self.atom_coords = []
        self.atomic_number = []
        self.atomic_nelec = []
        self.nelec = 0
        
        if self.unit not in ['angs', 'bohr']:
            raise ValueError('unit should be angs or bohr')

        self.process_atom_str()

        self.calculator = calculator(self.atoms, 
                                     self.atom_coords, 
                                     basis, scf, 
                                     self.unit)

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
            self.atom_coords.append([x*conv2bohr, y*conv2bohr, z*conv2bohr])

            self.atomic_number.append(element(atom_data[0]).atomic_number)
            self.atomic_nelec.append(element(atom_data[0]).electrons)
            self.nelec += element(atom_data[0]).electrons

        # size of the system
        self.natom = len(self.atoms)
        assert self.nelec % 2 == 0, "Only systems with equal up/down electrons allowed so far"
        self.nup = math.ceil(self.nelec/2)
        self.ndown = math.floor(self.nelec/2)

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
            method str -- 'center'  : all electron at the center of the molecule
                          'uniform' : all electrons in a box covering the molecule
                          'normal   : all electrons in a shpere covering the molecule 
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
            domain['sigma'] = np.diag(np.std(self.atom_coords, 0)+0.25)

        elif method == 'atomic':
            domain['atom_coords'] = self.atom_coords
            domain['atom_num'] = self.atomic_number
            domain['atom_nelec'] = self.atomic_nelec

        else:
            raise ValueError('Method to initialize the walkers not recognized')

        return domain


if __name__ == "__main__":
    from deepqmc.wavefunction.calculator.pyscf import CalculatorPySCF
    mol = Molecule(atom='Li 0 0 0; H 0 0 3.015',
                   calculator=CalculatorPySCF,
                   basis='dzp',
                   unit='bohr')
    
    
