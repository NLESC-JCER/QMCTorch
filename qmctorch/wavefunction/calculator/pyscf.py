import numpy as np
import os
from pyscf import gto, scf
import h5py
from types import SimpleNamespace
import numpy as np

from .calculator_base import CalculatorBase


class CalculatorPySCF(CalculatorBase):

    def __init__(self, atoms, atom_coords, basis, scf, units, molname):

        CalculatorBase.__init__(
            self, atoms, atom_coords, basis, scf, units, molname, 'pyscf')

    def run(self):
        """Run the scf calculation using PySCF."""

        # refresh the atom positions if necessary
        atom_str = self.get_atoms_str()

        # pyscf calculation
        mol = gto.M(
            atom=atom_str,
            basis=self.basis_name,
            unit=self.units)
        rhf = scf.RHF(mol).run()

        #self.save_data(mol, rhf)
        basis = self.get_basis_data(mol, rhf)
        return basis

        # print total energy
        # self.print_total_energy()

    def get_basis_data(self, mol, rhf):
        """Save the data to HDF5

        Arguments:
            mol {pyscf.gto.M} -- psycf Molecule
            rhf {pyscf.scf} -- scf object
        """

        kx = {0: [0], 1: [1, 0, 0], 2: [2, 1, 1, 0, 0, 0]}
        ky = {0: [0], 1: [0, 1, 0], 2: [0, 1, 0, 2, 1, 0]}
        kz = {0: [0], 1: [0, 0, 1], 2: [0, 0, 1, 0, 1, 2]}

        basis = SimpleNamespace()
        basis.TotalEnergy = rhf.e_tot
        basis.radial_type = 'gto'
        basis.harmonics_type = 'cart'

        # number of unique ao (e.g. px,py,px -> p)
        # h5['nbas'] = mol.nbas

        # total number of ao
        # wrong if there are d orbitals as counts only 5 d orbs (sph)
        # h5['nao'] = mol.nao

        basis.nao = len(mol.cart_labels())
        basis.nmo = rhf.mo_energy.shape[0]

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

        basis.nshells = nshells
        basis.index_ctr = index_ctr

        basis.bas_coeffs = np.array(bas_coeff)
        basis.bas_exp = np.array(bas_exp)
        basis.bas_norm = np.array(bas_norm)

        basis.bas_n = bas_n
        basis.bas_l = bas_l
        basis.bas_kr = np.array(bas_kr)

        basis.bas_kx = np.array(bas_kx)
        basis.bas_ky = np.array(bas_ky)
        basis.bas_kz = np.array(bas_kz)

        # molecular orbitals
        basis.mos = mol.cart2sph_coeff() @ rhf.mo_coeff
        basis.mos = self.normalize_columns(basis.mos)

        # atom coords
        basis.atom_coords_internal = self.atom_coords

        return basis

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
