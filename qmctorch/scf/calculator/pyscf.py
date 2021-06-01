from types import SimpleNamespace
import itertools
import numpy as np
from pyscf import gto, scf, dft
import shutil
from .calculator_base import CalculatorBase


class CalculatorPySCF(CalculatorBase):

    def __init__(self, atoms, atom_coords, basis, scf, units, molname, savefile):

        CalculatorBase.__init__(
            self, atoms, atom_coords, basis, scf, units, molname, 'pyscf', savefile)

    def run(self):
        """Run the scf calculation using PySCF."""

        # refresh the atom positions if necessary
        atom_str = self.get_atoms_str()

        # pyscf calculation
        mol = gto.M(
            atom=atom_str,
            basis=self.basis_name,
            unit=self.units,
            cart=False)

        if self.scf.lower() == 'hf':
            pyscf_data = scf.RHF(mol).run()

        elif self.scf.lower() == 'dft':
            pyscf_data = dft.RKS(mol)
            pyscf_data.xc = 'lda, vwn'
            pyscf_data = pyscf_data.newton()
            pyscf_data.kernel()

        if self.savefile:
            save_file_name = self.molname + '_pyscf.chkfile'
            shutil.copyfile(pyscf_data.chkfile, save_file_name)
            self.savefile = save_file_name

        basis = self.get_basis_data(mol, pyscf_data)
        return basis

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
        basis.radial_type = 'gto_pure'
        basis.harmonics_type = 'cart'

        # number of AO / MO
        # can be different if d or f orbs are present
        # due to cart 2 sph harmonics
        basis.nao = len(mol.cart_labels())
        basis.nmo = rhf.mo_energy.shape[0]

        # nshells is the number of bas per atoms
        nshells = [0] * mol.natm

        # init bas properties
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

        basis.nshells = nshells
        basis.index_ctr = index_ctr
        intervals = np.concatenate(([0], np.cumsum(nshells)))

        basis.nao_per_atom = []
        for i in range(len(intervals)-1):
            s, e = intervals[i], intervals[i+1]
            nao = len(np.unique(basis.index_ctr[s:e]))
            basis.nao_per_atom.append(nao)

        # determine the number of contraction per
        # atomic orbital
        basis.nctr_per_ao = np.array(
            [len(list(y)) for _, y in itertools.groupby(index_ctr)])

        basis.bas_coeffs = np.array(bas_coeff)
        basis.bas_exp = np.array(bas_exp)
        basis.bas_norm = np.array(bas_norm)

        basis.bas_n = bas_n
        basis.bas_l = bas_l

        # the cartesian gto are all :
        #   x^a y^b z^c exp(-zeta r)
        # i.e. there is no r dependency
        basis.bas_kr = np.zeros_like(basis.bas_exp)

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
