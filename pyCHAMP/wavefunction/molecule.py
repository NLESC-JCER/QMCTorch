import os
import numpy as np
from mendeleev import element
from pyscf import gto,scf

class Molecule(object):

    def __init__(self,atom=None,basis=''):

        self.atoms_str = atom
        self.basis = basis

        if self.basis.lower() not in ['sz','dz']:
            raise ValueError("Only DZ and SZ basis set supported")

        # process the atom name/pos
        self.atoms = []
        self.atom_coords = []
        self.nelec = 0
        self.process_atom_str()

        # get the basis folder
        self.basis_path = os.path.dirname(os.path.realpath(__file__))
        self.basis_path = os.path.join(self.basis_path,'atomicdata')
        self.basis_path = os.path.join(self.basis_path, basis.upper())

        # init the basis data
        self.nshells = []
        self.bas_exp = []
        self.bas_n = []
        self.bas_l = []
        self.bas_m = []

        # utilities dict for extracting data
        self.get_l = {'S':0,'P':1,'D':2}
        self.mult_bas = {'S':1,'P':3,'D':5}
        self.get_m = {'S':[0],'P':[-1,0,1],'D':[-2,-1,0,1,2]}

        # read/process basis info
        self.process_basis()

    def process_atom_str(self):

        atoms = self.atoms_str.split(';')
        for a in atoms:
            atom_data = a.split()
            self.atoms.append(atom_data[0])
            x,y,z = float(atom_data[1]),float(atom_data[2]),float(atom_data[3])
            self.atom_coords.append([x,y,z])
            self.nelec += element(atom_data[0]).electrons
        self.natom = len(self.atoms)

    def process_basis(self):

        # loop over all the atoms
        for at in self.atoms:

            self.nshells.append(0)
            
            # read the atom file
            fname = os.path.join(self.basis_path,at)
            with open(fname,'r') as f:
                data = f.readlines()

            # loop over all the basis
            for ibas in  range(data.index('BASIS\n')+1,data.index('END\n')):
                
                # split the data
                bas = data[ibas].split()
                bas_name = bas[0]
                zeta = float(bas[1])

                # get the primary quantum number
                n = int(bas_name[0])-1

                # secondary qn and multiplicity
                l = self.get_l[bas_name[1]]
                mult = self.mult_bas[bas_name[1]]

                # store the quantum numbers
                self.bas_n += [n]*mult
                self.bas_l += [l]*mult
                self.bas_m += self.get_m[bas_name[1]]

                # store the exponents
                self.bas_exp += [zeta]*mult

                # number of shells
                self.nshells[-1] += mult

        self.norb = np.sum(self.nshells)
        if self.basis.lower() == 'dz':
            self.norb = int(self.norb/2)
                
    def get_mo_coeffs(self,code='pyscf'):

        if code.lower() not in ['pyscf']:
            raise ValueError(code + 'not currently supported')

        if code.lower() == 'pyscf':
            mo = self._get_mo_pyscf()

        return mo

    def _get_mo_pyscf(self):

        pyscf_basis = {'sz':'sto-3g','dz':'dz'}
        mol = gto.M(atom=self.atoms_str,basis=pyscf_basis[self.basis])
        rhf = scf.RHF(mol).run()
        return self._normalize_columns(rhf.mo_coeff)

    @staticmethod
    def _normalize_columns(mat):
        norm = np.sqrt( (mat**2).sum(0) )
        return mat / norm


if __name__ == "__main__":

    m = Molecule(atom='H 0 0 0; O 0 0 0',basis='sz')

    
    


