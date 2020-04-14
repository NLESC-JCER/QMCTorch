import numpy as np
import os
from pyscf import gto, scf
import basis_set_exchange as bse
import json
import h5py

from deepqmc.wavefunction.calculator.calculator_base import CalculatorBase


class CalculatorPySCF(CalculatorBase):

    def __init__(self, atoms, atom_coords, basis, scf, units):

        CalculatorBase.__init__(
            self, atoms, atom_coords, basis, scf, units)
        self.basis.harmonics_type = 'cartesian'
        self.run()

    def run(self):
        """Run the scf calculation using PySCF."""

        h5name = ''.join(self.atoms) + '_' + self.basis_name + '.hdf5'
        if os.path.isfile(h5name):

            print('Reusing previous calculation from ', h5name)
            h5 = h5py.File(h5name, 'r')
            e = h5['TotalEnergy'][()]

        else:

            # refresh the atom positions if necessary
            atom_str = self.get_atoms_str()

            # pyscf calculation
            mol = gto.M(
                atom=atom_str,
                basis=self.basis_name,
                unit=self.units)
            rhf = scf.RHF(mol).run()

            self.save_data(mol, rhf, h5name)

        self.out_file = h5name

    def save_data(self, mol, rhf, file_name):
        """Save the data to HDF5

        Arguments:
            mol {pyscf.gto.M} -- psycf Molecule
            rhf {pyscf.scf} -- scf object
            file_name {str} -- name of the file
        """

        mval = {0: [0], 1: [-1, 1, 0], 2: [-2, -1, 0, 1, 2]}
        kx = {0: [0], 1: [1, 0, 0], 2: [2, 1, 1, 0, 0, 0]}
        ky = {0: [0], 1: [0, 1, 0], 2: [0, 1, 0, 2, 1, 0]}
        kz = {0: [0], 1: [0, 0, 1], 2: [0, 0, 1, 0, 1, 2]}

        h5 = h5py.File(file_name, 'w')
        h5['TotalEnergy'] = rhf.e_tot
        # number of unique ao (e.g. px,py,px -> p)
        h5['nbas'] = mol.nbas

        # total number of ao
        # wrong if there are d orbitals as counts only 5 d orbs (sph)
        # h5['nao'] = mol.nao
        h5['nao'] = len(mol.cart_labels())

        nshells = [0] * mol.natm

        bas_coeff, bas_exp = [], []
        index_ctr = []
        bas_n, bas_l, bas_m = [], [], []
        bas_kx, bas_ky, bas_kz = [], [], []
        bas_n = []
        bas_n_ori = self.get_bas_n(mol)

        iao = 0
        for ibas in range(mol.nbas):

            # number of contracted gaussian in that bas
            # nctr = mol.bas_nctr(ibas)
            nctr = mol.bas_nprim(ibas)

            # number of ao from that bas
            mult = mol.bas_len_cart(ibas)

            # quantum numbers
            n = bas_n_ori[ibas]
            l = mol.bas_angular(ibas)

            # get qn per bas
            bas_n += [n] * nctr * mult
            bas_l += [l] * nctr * mult

            # coeffs/exp
            bas_coeff += mol.bas_ctr_coeff(
                ibas).flatten().tolist() * mult
            bas_exp += mol.bas_exp(ibas).flatten().tolist() * mult

            # number of shell per atoms
            nshells[mol.bas_atom(ibas)] += nctr * mult

            for m in range(mult):
                index_ctr += [iao] * nctr
                iao += 1

            for k in kx[l]:
                bas_kx += [k] * nctr

            for k in ky[l]:
                bas_ky += [k] * nctr

            for k in kz[l]:
                bas_kz += [k] * nctr

        bas_norm = []
        for expnt, l in zip(bas_exp, bas_l):
            bas_norm.append(mol.gto_norm(l, expnt))

        bas_kr = np.array(bas_n) - np.array(bas_l) - 1

        h5.create_dataset('nshells', data=nshells)
        h5.create_dataset('index_ctr', data=index_ctr)

        h5.create_dataset('bas_coeff', data=bas_coeff)
        h5.create_dataset('bas_exp', data=bas_exp)
        h5.create_dataset('bas_norm', data=bas_norm)

        h5.create_dataset('bas_n', data=bas_n)
        h5.create_dataset('bas_l', data=bas_l)
        h5.create_dataset('bas_kr', data=bas_kr)

        h5.create_dataset('bas_kx', data=bas_kx)
        h5.create_dataset('bas_ky', data=bas_ky)
        h5.create_dataset('bas_kz', data=bas_kz)

        h5['nmo'] = rhf.mo_energy.shape[0]
        h5.create_dataset('mos', data=rhf.mo_coeff)
        h5.create_dataset('cart2sph', data=mol.cart2sph_coeff())

        h5.close()

    def get_basis(self):
        """Get the basis information needed to compute the AO values."""

        h5 = h5py.File(self.out_file, 'r')

        self.basis.radial_type = 'gto'
        self.basis.harmonics_type = 'cart'

        self.basis.nao = int(h5['nao'][()])
        self.basis.nmo = int(h5['nmo'][()])

        self.basis.nshells = h5['nshells'][()]

        self.basis.bas_kr = h5['bas_kr'][()]
        self.basis.bas_kx = h5['bas_kx'][()]
        self.basis.bas_ky = h5['bas_ky'][()]
        self.basis.bas_kz = h5['bas_kz'][()]

        self.basis.bas_n = h5['bas_n'][()]
        self.basis.bas_l = h5['bas_l'][()]

        self.basis.bas_exp = h5['bas_exp'][()]
        self.basis.bas_coeffs = h5['bas_coeff'][()]

        # the norm given by pyscf only accounts
        # for the radial part
        # self.basis.bas_norm = h5['bas_norm'][()]

        self.basis.index_ctr = h5['index_ctr'][()]
        self.basis.atom_coords_internal = self.atom_coords

        h5.close()

        self.check_basis(self.basis)

        return self.basis

    def get_mo_coeffs(self):
        """Get the MO coefficient expressed in the BAS."""

        h5 = h5py.File(self.out_file, 'r')

        nao = h5['nao'][()]
        nao = h5['nmo'][()]
        mos = h5['mos'][()]
        cart2sph = h5['cart2sph']
        bas_mos = cart2sph @ mos
        h5.close()

        return self.normalize_columns(bas_mos)

    def get_atoms_str(self):
        """Refresh the atom string.  Necessary when atom positions have changed. """
        atoms_str = ''
        natom = len(self.atoms)

        for iA in range(natom):
            atoms_str += self.atoms[iA] + ' '
            atoms_str += ' '.join(str(xi)
                                  for xi in self.atom_coords[iA])
            atoms_str += ';'
        return atoms_str

    @staticmethod
    def get_bas_n(mol):

        label2int = {'s': 1, 'p': 2, 'd': 3}
        labels = [l[:3] for l in mol.cart_labels(fmt=False)]
        unique_labels = []
        for l in labels:
            if l not in unique_labels:
                unique_labels.append(l)
        nlabel = [l[2][1] for l in unique_labels]
        n = [label2int[nl] for nl in nlabel]
        return n

    def parse_basis(self):
        """Get the properties of all the orbitals in the molecule."""

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
            for ishell, shell in enumerate(
                    data['elements'][str(at_num)]['electron_shells']):

                # get the primary number
                n = ishell

                # loop over the angular momentum
                for iangular, angular in enumerate(
                        shell['angular_momentum']):

                    # secondary qn and multiplicity
                    l = angular
                    mult = self.mult_bas[self.get_label[angular]]
                    nbas = len(shell['exponents'])
                    mvals = self.get_m[self.get_label[angular]]

                    for imult in range(mult):

                        self.norb += 1

                        # store coeffs and exps of the bas
                        self.bas_exp += (
                            np.array(
                                shell['exponents']).astype('float')).tolist()
                        self.bas_coeffs += np.array(
                            shell['coefficients'][iangular]).astype('float').tolist()

                        # index of the contraction
                        self.index_ctr += [self.norb - 1] * nbas

                        # store the quantum numbers
                        self.bas_n += [n] * nbas
                        self.bas_l += [l] * nbas
                        self.bas_m += [mvals[imult]] * nbas

                    # number of shells
                    self.nshells[-1] += mult * nbas
