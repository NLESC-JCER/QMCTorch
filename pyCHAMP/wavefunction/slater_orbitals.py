import torch
from torch import nn
import torch.nn.functional as F
from math import pi as PI

from pyCHAMP.wavefunction.spherical_harmonics import SphericalHarmonics

class STO(nn.Module):

    def __init__(self,
                nelec,
                atom_coords,
                nshells,
                bas_n,
                bas_l,
                bas_m, 
                bas_exp,
                basis ):

        '''Radial Basis Function Layer in N dimension

        Args:
            nelec : number of electron
            atom_coords : position of the atoms
            nshells : number of bas per atom
            bas_n : 1st quantum number of the bas
            bas_l : 2nd quantum number of the bas
            bas_m : 3rd quantum number of the bas
            bas_exp : exponent of the bas
            basis : type of the basis used (dz or sz)
        '''

        super(STO,self).__init__()

        # wavefunction data
        self.nelec = nelec
        self.ndim = 3

        # make the atomic position optmizable
        self.atom_coords = nn.Parameter(torch.tensor(atom_coords))
        self.atom_coords.requires_grad = True
        self.natoms = len(self.atom_coords)

        # define the BAS positions
        self.nshells = torch.tensor(nshells)
        self.bas_coords = self.atom_coords.repeat_interleave(self.nshells,dim=0)
        self.nbas = len(self.bas_coords)

        # get the exponents of the bas
        self.bas_exp = nn.Parameter(torch.tensor(bas_exp))
        self.bas_exp.requires_grad = True

        # get the quantum number
        self.bas_n = torch.tensor(bas_n).float()
        self.bas_l = torch.tensor(bas_l)
        self.bas_m = torch.tensor(bas_m)

        # basis
        self.basis = basis
        if self.basis.lower() not in ['sz','dz']:
            raise ValueError("Only DZ and SZ basis set supported")

    def forward(self,input):
        
        # get the pos of the bas
        self.bas_coords = self.atom_coords.repeat_interleave(self.nshells,dim=0)

        # get the x,y,z, distance component of each point from each RBF center
        # -> (Nbatch,Nelec,Nrbf,Ndim)
        xyz =  (input.view(-1,self.nelec,1,self.ndim) - self.bas_coords[None,...])
        
        # compute the distance
        # -> (Nbatch,Nelec,Nrbf)
        R = torch.sqrt((xyz**2).sum(3))
        
        # radial part
        # -> (Nbatch,Nelec,Nrbf)
        X = R**self.bas_n * torch.exp(-self.bas_exp*R)
        
        # compute by the spherical harmonics
        # -> (Nbatch,Nelec,Nrbf)
        Y = SphericalHarmonics(xyz,self.bas_l,self.bas_m)
        
        # product
        XY = X * Y

        # add the components if DZ basis set
        if self.basis == 'dz':
            nrbf = XY.shape[-1]
            norb = int(nrbf/2)
            XY = 0.5*XY.view(-1,self.nelec,2,norb).sum(2)

        return XY

if __name__ == "__main__":

    from pyCHAMP.wavefunction.molecule import Molecule

    m = Molecule(atom='H 0 0 0; H 0 0 1',basis='sz')

    sto = STO(nelec=m.nelec,
                atom_coords=m.atom_coords,
                nshells=m.nshells,
                bas_n=m.bas_n,
                bas_l=m.bas_l,
                bas_m=m.bas_l, 
                bas_exp=m.bas_exp,
                basis=m.basis)

    pos = torch.rand(20,sto.nelec*3)
    aoval = sto.forward(pos)