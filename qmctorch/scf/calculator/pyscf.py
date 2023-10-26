from types import SimpleNamespace
import itertools
import numpy as np
from pyscf import gto, scf, dft
import shutil
from .calculator_base import CalculatorBase
from ... import log


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
            unit='Bohr',
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

        # sphereical quantum nummbers
        mvalues = {0: [0], 1: [-1,0,1], 2: [-2,-1,0,1,2]}

        # cartesian quantum numbers
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
        bas_n, bas_m, bas_l = [], [], []
        bas_zeta = []
        bas_kx, bas_ky, bas_kz = [], [], []
        bas_n_ori = self.get_bas_n(mol)

        iao = 0
        ishell = 0
        for ibas in range(mol.nbas):

            # number of contracted gto per shell
            nctr = mol.bas_nctr(ibas)

            # number of primitive gaussian in that shell
            nprim = mol.bas_nprim(ibas)

            # number of cartesian component of that bas ?
            ncart_comp = mol.bas_len_cart(ibas)

            # quantum numbers
            n = bas_n_ori[ibas]
            lval = mol.bas_angular(ibas)

            # coeffs and exponents
            coeffs = mol.bas_ctr_coeff(ibas)
            exps =  mol.bas_exp(ibas)

            # deal with  multiple zeta
            if coeffs.shape != (nprim, nctr):
                raise ValueError('Contraction coefficients issue')
         
            ictr = 0
            while ictr < nctr:

                n = bas_n_ori[ishell]
                coeffs_ictr = coeffs[:,ictr] / (ictr+1)

                # coeffs/exp
                bas_coeff += coeffs_ictr.flatten().tolist() * ncart_comp
                bas_exp += exps.flatten().tolist() * ncart_comp

                # get quantum numbers per bas
                bas_n += [n] * nprim * ncart_comp
                bas_l += [lval] * nprim * ncart_comp

                # record the zetas per bas
                bas_zeta += [nctr] * nprim * ncart_comp

                # number of shell per atoms
                nshells[mol.bas_atom(ibas)] += nprim * ncart_comp

                for _ in range(ncart_comp):
                    index_ctr += [iao] * nprim
                    iao += 1

                for m in mvalues[lval]:
                    bas_m += [m] * nprim

                for k in kx[lval]:
                    bas_kx += [k] * nprim

                for k in ky[lval]:
                    bas_ky += [k] * nprim

                for k in kz[lval]:
                    bas_kz += [k] * nprim

                ictr += 1
                ishell += 1

        # normalize the basis function
        bas_norm = []
        for expnt, lval in zip(bas_exp, bas_l):
            bas_norm.append(mol.gto_norm(lval, expnt))

        # load in data structure
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
        basis.bas_m = bas_m

        # the cartesian gto are all: x^a y^b z^c exp(-zeta r)
        # i.e. there is no kr dependency
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

        recognized_labels = ['s','p','d']

        label2int = {'s': 1, 'p': 2, 'd': 3}
        labels = [l[:3] for l in mol.cart_labels(fmt=False)]
        unique_labels = []
        for l in labels:
            if l not in unique_labels:
                unique_labels.append(l)
        nlabel = [l[2][1] for l in unique_labels]

        if np.any([nl not in recognized_labels for nl in nlabel]):
            log.error('the pyscf calculator only supports the following orbitals: {0}', recognized_labels)
            log.error('The following orbitals have been found: {0}', nlabel)
            log.error('Using the basis set: {0}', mol.basis)
            raise ValueError('Basis set not supported')

        n = [label2int[nl] for nl in nlabel]
        return n
