import os
import numpy as np

class Molecule(object):

    def __init__(self,atom=None,basis=''):

        self.atoms_str = atom
        self.basis = basis

        self.atoms = []
        self.atom_coords = []
        self.process_atom_str()

        __path__ = './atomicdata'        
        self.basis_path = os.path.join( __path__, basis.upper())
        self.nshells = []
        self.bas_exp = []
        self.bas_n = []
        self.bas_l = []
        self.bas_m = []

        self.get_l = {'S':0,'P':1,'D':2}
        self.mult_bas = {'S':1,'P':3,'D':5}
        self.get_m = {'S':[0],'P':[-1,0,1],'D':[-2,-1,0,1,2]}

        self.process_basis()

    def process_atom_str(self):

        atoms = self.atoms_str.split(';')
        for a in atoms:
            atom_data = a.split()
            self.atoms.append(atom_data[0])
            x,y,z = float(atom_data[1]),float(atom_data[2]),float(atom_data[3])
            self.atom_coords.append([x,y,z])

    def process_basis(self):

        # loop over all the atoms
        for at in self.atoms:
            
            # read the atom file
            fname = os.path.join(self.basis_path,at)
            with open(fname,'r') as f:
                data = f.readlines()

            # loop over all the basis
            for ibas in  range(data.index('BASIS\n')+1,data.index('END\n')):
                
                # split the data
                d = data[ibas].split()
                print(d)
                # get the primary quantum number
                n = int(d[0][0])-1

                # secondary qn and multiplicity
                l = self.get_l[d[0][1]]
                mult = self.mult_bas[d[0][1]]

                # store the quantum numbers
                self.bas_n += [n]*mult
                self.bas_l += [l]*mult
                self.bas_m += self.get_m[d[0][1]]

                # store the exponets
                zeta = float(d[1])
                self.bas_exp += [zeta]*mult

if __name__ == "__main__":

    m = Molecule(atom='H 0 0 0; O 0 0 0',basis='sz')

    
    


