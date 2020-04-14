import numpy as np
import os
import shutil
import warnings
from .calculator_base import CalculatorBase

try:
    from scm import plams
except ModuleNotFoundError:
    warnings.warn('scm python module not found')


class CalculatorADF(CalculatorBase):

    def __init__(self, atoms, atom_coords, basis, scf, units):

        CalculatorBase.__init__(
            self, atoms, atom_coords, basis, scf, units)
        self.run()

    def run(self):
        """Run the calculation

        Raises:
            ValueError: [description]

        Returns:
            np.ndarray -- molecular orbital matrix
        """

        wd = ''.join(self.atoms) + '_' + self.basis_name
        t21_name = wd + '.t21'
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

        self.basis.nao = kf.read('Basis', 'naos')
        self.basis.nmo = kf.read('A', 'nmo_A')

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

        self.basis.nshells = []
        self.basis.bas_kx, self.basis.bas_ky, self.basis.bas_kz = [], [], []
        self.basis.bas_kr = []
        self.basis.bas_exp, self.basis.bas_norm = [], []

        for iat, at in enumerate(atom_type):

            number_copy = nqptr[iat + 1] - nqptr[iat]
            idx_bos = list(range(nbptr[iat] - 1, nbptr[iat + 1] - 1))

            self.basis.nshells += [nshells[iat]] * number_copy

            self.basis.bas_kx += list(bas_kx[idx_bos]) * number_copy
            self.basis.bas_ky += list(bas_ky[idx_bos]) * number_copy
            self.basis.bas_kz += list(bas_kz[idx_bos]) * number_copy
            self.basis.bas_kr += list(bas_kr[idx_bos]) * number_copy

            self.basis.bas_exp += list(bas_exp[idx_bos]) * number_copy
            self.basis.bas_norm += list(
                bas_norm[idx_bos]) * number_copy

        self.basis.nshells = np.array(self.basis.nshells)
        self.basis.bas_kx = np.array(self.basis.bas_kx)
        self.basis.bas_ky = np.array(self.basis.bas_ky)
        self.basis.bas_kz = np.array(self.basis.bas_kz)

        self.basis.bas_kr = np.array(self.basis.bas_kr)
        self.basis.bas_exp = np.array(self.basis.bas_exp)
        self.basis.bas_coeffs = np.ones_like(self.basis.bas_exp)
        self.basis.bas_norm = np.array(self.basis.bas_norm)

        self.basis.index_ctr = np.arange(self.basis.nao)
        self.basis.atom_coords_internal = np.array(
            kf.read('Geometry', 'xyz')).reshape(-1, 3)

        self.check_basis(self.basis)

        return self.basis

    def get_mo_coeffs(self):
        """Get the MO coefficient expressed in the BAS."""

        kf = plams.KFFile(self.out_file)
        nao = kf.read('Basis', 'naos')
        nmo = kf.read('A', 'nmo_A')
        self.mos = np.array(kf.read('A', 'Eigen-Bas_A'))
        self.mos = self.mos.reshape(nmo, nao).T
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
                for iangular, angular in enumerate(
                        shell['angular_momentum']):

                    # secondary qn and multiplicity
                    lval = angular
                    if lval > self.max_angular:
                        raise ValueError('Only orbital up to l=%d (%s) are \
                                         currently supported',
                                         self.max_angular,
                                         self.get_label[self.max_angular])

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
                        self.bas_n += [n] * nbas
                        self.bas_l += [lval] * nbas
                        self.bas_m += [mvals[imult]] * nbas

                    # number of shells
                    self.nshells[-1] += nbas * mult

        self.index_ctr = list(range(self.norb))
