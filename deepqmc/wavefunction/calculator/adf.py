import numpy as np 
import os
import shutil
from scm import plams
import json
from deepqmc.wavefunction.calculator.calculator_base import CalculatorBase

class CalculatorADF(CalculatorBase):

    def __init__(self, atoms, atom_coords, basis, scf, units):

        CalculatorBase.__init__(self,atoms, atom_coords, basis, scf, units)
        self.run()
        

    def run(self):
        """Run the calculation

        Raises:
            ValueError: [description]

        Returns:
            np.ndarray -- molecular orbital matrix
        """

        wd = ''.join(self.atoms)+'_'+self.basis_name
        t21_name = wd+'.t21'
        plams_wd = './plams_workdir'
        t21_path = os.path.join(plams_wd, os.path.join(wd, t21_name))

        if os.path.isfile(t21_name):

            print('Reusing previous calculation from ', t21_name)
            kf = plams.KFFile(t21_name)
            e = kf.read('Total Energy', 'Total energy')

        else:

            # init PLAMS
            plams.init()
            plams.config.log.stdout = -1
            plams.config.erase_workdir = True

            # create the molecule
            mol = plams.Molecule()
            for at, xyz in zip(self.atoms, self.atom_coords):
                mol.add_atom(plams.Atom(symbol=at, coords=tuple(xyz)))

            # settings in PLAMS
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

            # run the ADF job
            job = plams.ADFJob(molecule=mol, settings=sett, name=wd)
            job.run()

            # read the energy from the t21 file
            e = job.results.readkf('Total Energy', 'Total energy')

            # make a copy of the t21 file
            shutil.copyfile(t21_path, t21_name)
            shutil.rmtree(plams_wd)

        # print energy
        print('== SCF Energy : ', e)
        self.out_file = t21_name

    def get_basis(self):
        """Get the basis information needed to compute the AO values."""
        
        kf = plams.KFFile(self.out_file)

        self.basis.radial_type = 'sto'
        self.basis.harmonics_type = 'cart'

        self.basis.nao  = kf.read('Basis','naos')
        self.basis.nmo = kf.read('A', 'nmo_A')

        nbptr = np.array(kf.read('Basis','nbptr')) 
        self.basis.nshells = np.array([nbptr[i]-nbptr[i-1] for i in range(1,len(nbptr))])

        self.basis.bas_kx = np.array(kf.read('Basis', 'kx'))
        self.basis.bas_ky = np.array(kf.read('Basis', 'ky'))
        self.basis.bas_kz = np.array(kf.read('Basis', 'kz'))

        self.basis.bas_n = np.array(kf.read('Basis', 'kr'))

        self.basis.bas_exp = np.array(kf.read('Basis', 'alf'))
        self.basis.bas_coeffs = np.ones_like(self.basis.bas_exp)
        self.basis.bas_norm = np.array(kf.read('Basis', 'bnorm'))

        self.basis.index_ctr = np.arange(self.basis.nao)
        return self.basis 

    def get_mos(self):
        """Get the MO coefficient expressed in the BAS."""

        kf = plams.KFFile(self.out_file)
        nao  = kf.read('Basis','naos')
        nmo = kf.read('A', 'nmo_A')
        self.mos = np.array(kf.read('A', 'Eigen-Bas_A'))
        self.mos = self.mos.reshape(nao, nmo).T
        self.mos = self.normalize_columns(self.mos)
        return self.mos


    def parse_basis(self):

        """Get the properties of all orbital in the molecule.

        Raises:
            ValueError: if orbitals larger than D are used.
        """

        # number of orbs
        self.basis.naos = 0

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

                        # self.norb += 1
                        self.norb += nbas

                        # store coeffs and exps of the bas
                        self.bas_exp += shell['exponents'][iangular]
                        self.bas_coeffs += shell['coefficients'][iangular]

                        # store the quantum numbers
                        self.bas_n += [n]*nbas
                        self.bas_l += [l]*nbas
                        self.bas_m += [mvals[imult]]*nbas

                    # number of shells
                    self.nshells[-1] += nbas*mult

        self.index_ctr = list(range(self.norb))

    def _get_atomic_data(self, at):
        """Get the properties of an STO basis of a single atom

        Arguments:
            at {str} -- atom type

        Raises:
            ValueError: if orbital up thad D are used

        Returns:
            dict -- properties of the different orbital for the specified atom
        """

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