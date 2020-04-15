import numpy as np
import os
from pyscf import gto, scf
import h5py

from .calculator_base import CalculatorBase


class CalculatorPySCF(CalculatorBase):

    def __init__(self, atoms, atom_coords, basis, scf, units, molname):

        CalculatorBase.__init__(
            self, atoms, atom_coords, basis, scf, units, molname, 'pyscf')
        self.run()

    def run(self):
        """Run the scf calculation using PySCF."""

        if os.path.isfile(self.out_file):
            print('Reusing previous calculation from ', self.out_file)

        else:

            # refresh the atom positions if necessary
            atom_str = self.get_atoms_str()

            # pyscf calculation
            mol = gto.M(
                atom=atom_str,
                basis=self.basis_name,
                unit=self.units)
            rhf = scf.RHF(mol).run()

            self.save_data(mol, rhf)

        # print total energy
        self.print_total_energy()

    def save_data(self, mol, rhf):
        """Save the data to HDF5

        Arguments:
            mol {pyscf.gto.M} -- psycf Molecule
            rhf {pyscf.scf} -- scf object
        """

        kx = {0: [0], 1: [1, 0, 0], 2: [2, 1, 1, 0, 0, 0]}
        ky = {0: [0], 1: [0, 1, 0], 2: [0, 1, 0, 2, 1, 0]}
        kz = {0: [0], 1: [0, 0, 1], 2: [0, 0, 1, 0, 1, 2]}

        h5 = h5py.File(self.out_file, 'w')

        h5['TotalEnergy'] = rhf.e_tot
        h5['radial_type'] = 'sto'
        h5['harmonics_type'] = 'cart'

        # number of unique ao (e.g. px,py,px -> p)
        # h5['nbas'] = mol.nbas

        # total number of ao
        # wrong if there are d orbitals as counts only 5 d orbs (sph)
        # h5['nao'] = mol.nao
        h5['nao'] = len(mol.cart_labels())
        h5['nmo'] = rhf.mo_energy.shape[0]

        nshells = [0] * mol.natm

        bas_coeff, bas_exp = [], []
        index_ctr = []
        bas_n, bas_l = [], []
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
            lval = mol.bas_angular(ibas)

            # get qn per bas
            bas_n += [n] * nctr * mult
            bas_l += [lval] * nctr * mult

            # coeffs/exp
            bas_coeff += mol.bas_ctr_coeff(
                ibas).flatten().tolist() * mult
            bas_exp += mol.bas_exp(ibas).flatten().tolist() * mult

            # number of shell per atoms
            nshells[mol.bas_atom(ibas)] += nctr * mult

            for m in range(mult):
                index_ctr += [iao] * nctr
                iao += 1

            for k in kx[lval]:
                bas_kx += [k] * nctr

            for k in ky[lval]:
                bas_ky += [k] * nctr

            for k in kz[lval]:
                bas_kz += [k] * nctr

        bas_norm = []
        for expnt, lval in zip(bas_exp, bas_l):
            bas_norm.append(mol.gto_norm(lval, expnt))

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

        # molecular orbitals
        mos = mol.cart2sph_coeff() @ rhf.mo_coeff
        mos = self.normalize_columns(mos)
        h5.create_dataset('mos', data=mos)

        # atom coords
        h5.create_dataset('atom_coords_internal',
                          data=self.atom_coords)
        h5.close()

        self.check_h5file()

    def get_atoms_str(self):
        """Refresh the atom string (use after atom move). """
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
