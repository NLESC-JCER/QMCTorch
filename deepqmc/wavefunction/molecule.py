import os
import math
import numpy as np
from mendeleev import element
from pyscf import gto, scf
import basis_set_exchange as bse
import json


class Molecule(object):

    def __init__(self, atom=None,
                 basis_type='sto',
                 basis='sz',
                 unit='bohr'):

        self.atoms_str = atom
        self.basis_type = basis_type.lower()
        self.basis = basis.lower()
        self.code_mo = 'pyscf'

        if self.basis_type not in ['sto', 'gto']:
            raise ValueError("basis_type must be sto or gto")

        # process the atom name/pos
        self.max_angular = 2
        self.atoms = []
        self.atom_coords = []
        self.atomic_number = []
        self.nelec = 0
        self.unit = unit

        if self.unit not in ['angs','bohr']:
            raise ValueError('unit should be angs or bohr')

        self.process_atom_str()
        self.get_bonds()

        # get the basis folder
        self.basis_path = os.path.dirname(os.path.realpath(__file__))
        self.basis_path = os.path.join(self.basis_path, 'atomicdata')
        self.basis_path = os.path.join(self.basis_path, self.basis_type)
        self.basis_path = os.path.join(self.basis_path, basis.upper())

        # init the basis data
        self.nshells = []  # number of shell per atom
        self.index_ctr = []  # index of the contraction
        self.bas_exp = []
        self.bas_coeffs = []
        self.bas_n = []
        self.bas_l = []
        self.bas_m = []

        # utilities dict for extracting data
        self.get_label = {0: 'S', 1: 'P', 2: 'D'}
        self.get_l = {'S': 0, 'P': 1, 'D': 2}
        self.mult_bas = {'S': 1, 'P': 3, 'D': 5}
        self.get_m = {'S': [0], 'P': [-1, 1, 0], 'D': [-2, -1, 0, 1, 2]}

        # for cartesian
        self.get_lmn_cart = {'S': [0, 0, 0],
                             'P': [[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]],
                             'D': [[2, 0, 0],
                                   [0, 2, 0],
                                   [0, 0, 2],
                                   [1, 1, 0],
                                   [1, 0, 1],
                                   [0, 1, 1]]}

        # read/process basis info
        self.process_basis()

    def process_atom_str(self):
        '''Process the input file.'''

        # if we have an xyz file
        if os.path.isfile(self.atoms_str):
            with open(self.atoms_str, 'r') as f:
                data = f.readlines()
            atoms = data[2:]
            self.atoms_str = ''
            for a in atoms[:-1]:
                self.atoms_str += a + '; '
            self.atoms_str += atoms[-1]

        # if we have a string
        else:
            atoms = self.atoms_str.split(';')

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
            self.nelec += element(atom_data[0]).electrons

        # size of the system
        self.natom = len(self.atoms)
        assert self.nelec % 2 == 0, "Only systems with equal up/down electrons allowed so far"
        self.nup = math.ceil(self.nelec/2)
        self.ndown = math.floor(self.nelec/2)

    def get_bonds(self):
        self.bonds = list()
        for iat1 in range(self.natom-1):
            at1 = self.atoms[iat1]
            xyz1 = np.array(self.atom_coords[iat1])
            for iat2 in range(iat1+1, self.natom):
                at2 = self.atoms[iat2]
                xyz2 = np.array(self.atom_coords[iat2])

                d = np.sqrt(np.sum((xyz1-xyz2)**2))
                thr = self._get_max_blength(at1, at2)
                if d < thr:
                    self.bonds.append((iat1, iat2))

    @staticmethod
    def _get_max_blength(at1, at2):
        bond = {('H', 'H'): 2.}
        if (at1, at2) in bond:
            return bond[(at1, at2)]
        else:
            return 2.

    def process_basis(self):
        if self.basis_type == 'sto':
            self._process_sto()

        elif self.basis_type == 'gto':
            self._process_gto()

    def _get_sto_atomic_data(self, at):

        atomic_data = {'electron_shells': {}}

        # read the atom file
        fname = os.path.join(self.basis_path, at)
        with open(fname, 'r') as f:
            data = f.readlines()

        # loop over all the basis
        for ibas in range(data.index('BASIS\n')+1, data.index('END\n')):

            # split the data
            bas = data[ibas].split()

            if len(bas) == 0:
                continue

            bas_name = bas[0]
            zeta = float(bas[1])

            # get the primary quantum number
            n = int(bas_name[0])-1

            if n not in atomic_data['electron_shells']:
                atomic_data['electron_shells'][n] = {'angular_momentum': [],
                                                     'exponents': [],
                                                     'coefficients': []}

            # secondary qn and multiplicity
            if bas_name[1] in self.get_l.keys():
                l = self.get_l[bas_name[1]]
            else:
                raise ValueError('Only orbital up to l=%d (%s) are currently supported',
                                 self.max_angular, self.get_label[self.max_angular])

            # store it
            if l not in atomic_data['electron_shells'][n]['angular_momentum']:
                atomic_data['electron_shells'][n]['angular_momentum'].append(l)
                atomic_data['electron_shells'][n]['coefficients'].append([])
                atomic_data['electron_shells'][n]['exponents'].append([])

            atomic_data['electron_shells'][n]['coefficients'][-1].append(1.)
            atomic_data['electron_shells'][n]['exponents'][-1].append(zeta)

        return atomic_data

    def _process_sto(self):

        # number of orbs
        self.norb = 0

        # loop over all the atoms
        for at in self.atoms:

            data = self._get_sto_atomic_data(at)
            self.nshells.append(0)

            for ishell, shell in data['electron_shells'].items():

                # primary quantum number
                n = ishell

                # loop over the angular momentum
                for iangular, angular in enumerate(shell['angular_momentum']):

                    # secondary qn and multiplicity
                    l = angular
                    if l > self.max_angular:
                        raise ValueError('Only orbital up to l=%d (%s) are currently supported',
                                         self.max_angular, self.get_label[self.max_angular])

                    mult = self.mult_bas[self.get_label[angular]]
                    nbas = len(shell['coefficients'][iangular])
                    mvals = self.get_m[self.get_label[angular]]

                    for imult in range(mult):

                        self.norb += 1

                        # store coeffs and exps of the bas
                        self.bas_exp += shell['exponents'][iangular]
                        self.bas_coeffs += shell['coefficients'][iangular]

                        # index of the contraction
                        self.index_ctr += [self.norb-1] * nbas

                        # store the quantum numbers
                        self.bas_n += [n]*nbas
                        self.bas_l += [l]*nbas
                        self.bas_m += [mvals[imult]]*nbas

                    # number of shells
                    self.nshells[-1] += nbas*mult

    def _process_gto(self):

        # number of orbs
        self.norb = 0

        # loop over all the atoms
        for at in self.atoms:

            at_num = element(at).atomic_number
            self.nshells.append(0)

            # import data
            data = json.loads(bse.get_basis(
                self.basis, elements=[at_num], fmt='JSON'))

            # loop over the electronic shells
            for ishell, shell in enumerate(data['elements'][str(at_num)]['electron_shells']):

                # get the primary number
                n = ishell

                # loop over the angular momentum
                for iangular, angular in enumerate(shell['angular_momentum']):

                    # secondary qn and multiplicity
                    l = angular
                    mult = self.mult_bas[self.get_label[angular]]
                    nbas = len(shell['exponents'])
                    mvals = self.get_m[self.get_label[angular]]

                    for imult in range(mult):

                        self.norb += 1

                        # store coeffs and exps of the bas
                        self.bas_exp += (np.array(shell['exponents']
                                                  ).astype('float')).tolist()
                        self.bas_coeffs += np.array(
                            shell['coefficients'][iangular]).astype('float').tolist()

                        # index of the contraction
                        self.index_ctr += [self.norb-1] * nbas

                        # store the quantum numbers
                        self.bas_n += [n]*nbas
                        self.bas_l += [l]*nbas
                        self.bas_m += [mvals[imult]]*nbas

                    # number of shells
                    self.nshells[-1] += mult*nbas

    def get_mo_coeffs(self, code='pyscf'):

        if code is None:
            code = self.code_mo
        else:
            self.code_mo = code

        if self.basis_type == 'gto':

            if code.lower() == 'pyscf':
                mo = self._get_mo_pyscf()
            else:
                raise ValueError(
                    code + 'not currently supported for GTO orbitals')

        elif self.basis_type == 'sto':
                
            if code.lower() == 'pyscf':
                mo = self._get_mo_pyscf()
            elif code.lower() == 'adf':
                mo = self._get_mo_adf()
            else:
                raise ValueError(
                    code + 'not currently supported for STO orbitals')


        return mo

    def _get_mo_pyscf(self):

        if self.basis_type == 'sto':
            bdict = {'sz': 'sto-3g', 'dz': 'sto-3g'}
            pyscf_basis = bdict[self.basis]
        else:
            pyscf_basis = self.basis

        self._get_atoms_str()

        mol = gto.M(atom=self.atoms_str, basis=pyscf_basis, unit=self.unit)
        rhf = scf.RHF(mol).run()
        return self._normalize_columns(rhf.mo_coeff)

    def _get_mo_adf(self):

        from scm import plams
        import shutil        

        wd = ''.join(self.atoms)+'_'+self.basis
        t21_name = wd+'.t21'
        plams_wd = './plams_workdir'
        t21_path = os.path.join(plams_wd,os.path.join(wd,t21_name))

        if os.path.isfile(t21_name):

            kf = plams.KFFile(t21_name)
            nmo = kf.read('A','nmo_A')
            mos = np.array(kf.read('A','Eigen-Bas_A'))

        else:

            plams.init()
            plams.config.log.stdout = -1
            plams.config.erase_workdir = True

            mol = plams.Molecule()
            for at,xyz in zip (self.atoms,self.atom_coords):
                mol.add_atom(plams.Atom(symbol=at,coords=tuple(xyz)))

            sett = plams.Settings()
            sett.input.basis.type = self.basis.upper()
            sett.input.basis.core = 'None'
            sett.input.symmetry = 'nosym'
            sett.input.XC.HartreeFock = ''

            if self.unit == 'angs':
                sett.input.units.length = 'Angstrom'
            elif self.unit == 'bohr':
                sett.input.units.length = 'Bohr'
            else:
                raise ValueError('unit should be angs or bohr')

            job = plams.ADFJob(molecule=mol,settings=sett,name=wd)
            job.run()

            nmo = job.results.readkf('A', 'nmo_A')
            mos = np.array(job.results.readkf('A', 'Eigen-Bas_A'))

            shutil.copyfile(t21_path,t21_name)

        mos = mos.reshape(nmo, nmo).T
        return  self._normalize_columns(mos)

    def _get_atoms_str(self):
        self.atoms_str = ''
        for iA in range(self.natom):
            self.atoms_str += self.atoms[iA] + ' '
            self.atoms_str += ' '.join(str(xi) for xi in self.atom_coords[iA])
            self.atoms_str += ';'

    def update_mo_coeffs(self):
        return self.get_mo_coeffs(code=None)

    @staticmethod
    def _normalize_columns(mat):
        norm = np.sqrt((mat**2).sum(0))
        return mat / norm

    def domain(self, method):

        domain = dict()

        if method == 'center':
            domain['center'] = np.mean(self.atom_coords, 0)

        elif method == 'uniform':
            domain['min'] = np.min(self.atom_coords) - 0.5
            domain['max'] = np.max(self.atom_coords) + 0.5

        elif method == 'normal':
            domain['mean'] = np.mean(self.atom_coords, 0)
            domain['sigma'] = np.diag(np.std(self.atom_coords, 0)+0.25)

        else:
            raise ValueError('Method to initialize the walkers not recognized')
        return domain


if __name__ == "__main__":
    mol = Molecule(atom='Li 0 0 0; H 0 0 3.015',
                   basis_type='sto',
                   basis='sz',
                   unit='bohr')
    mo = mol.get_mo_coeffs(code='adf')
    print(mo)