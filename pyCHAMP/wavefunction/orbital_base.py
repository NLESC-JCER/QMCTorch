import torch
from torch import nn
import torch.nn.functional as F
from math import pi as PI

from pyCHAMP.wavefunction.spherical_harmonics import SphericalHarmonics

class Orbitals(nn.Module):

    def __init__(self,
                basis_type,
                nelec,
                atom_coords,
                norb,
                nshells,
                index_ctr,
                bas_n,
                bas_l,
                bas_m, 
                bas_coeffs,
                bas_exp,
                basis ):

        '''Radial Basis Function Layer in N dimension

        Args:
            basis_type : sto or gto
            nelec : number of electron
            atom_coords : position of the atoms
            norb : total number of orbitals
            nshells : number of bas per atom
            bas_n : 1st quantum number of the bas
            bas_l : 2nd quantum number of the bas
            bas_m : 3rd quantum number of the bas
            bas_exp : exponent of the bas
        '''

        super(Orbitals,self).__init__()

        # wavefunction data
        self.nelec = nelec
        self.norb = norb
        self.ndim = 3

        # make the atomic position optmizable
        self.atom_coords = nn.Parameter(torch.tensor(atom_coords))
        self.atom_coords.requires_grad = True
        self.natoms = len(self.atom_coords)

        # define the BAS positions
        self.nshells = torch.tensor(nshells)
        self.bas_coords = self.atom_coords.repeat_interleave(self.nshells,dim=0)
        self.nbas = len(self.bas_coords)

        #index for the contractions
        self.index_ctr = torch.tensor(index_ctr)

        # get the coeffs of the bas
        self.bas_coeffs = torch.tensor(bas_coeffs)

        # get the exponents of the bas
        self.bas_exp = nn.Parameter(torch.tensor(bas_exp))
        self.bas_exp.requires_grad = True

        # get the quantum number
        self.bas_n = torch.tensor(bas_n).float()
        self.bas_l = torch.tensor(bas_l)
        self.bas_m = torch.tensor(bas_m)

        # select the radial aprt
        radial_dict = {'sto':self.radial_slater, 'gto':self.radial_gaussian}
        self.radial = radial_dict[basis_type]
        
    def radial_slater(self,R):
        return R**self.bas_n * torch.exp(-self.bas_exp*R)

    def radial_gaussian(self,R):
        return R**self.bas_n * torch.exp(-self.bas_exp*R**2)

    def forward(self,input):
        
        nbatch = input.shape[0]

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
        R = self.radial(R)
        
        # compute by the spherical harmonics
        # -> (Nbatch,Nelec,Nrbf)
        Y = SphericalHarmonics(xyz,self.bas_l,self.bas_m)
        
        # product with coefficients
        phi = self.bas_coeffs * R * Y

        # contract the basis
        psi = torch.zeros(nbatch,self.nelec,self.norb)
        psi.index_add_(2,self.index_ctr,phi)

        return psi




if __name__ == "__main__":

    from pyCHAMP.wavefunction.molecule import Molecule

    #m = Molecule(atom='H 0 0 0; H 0 0 1',basis_type='sto',basis='dz')
    m = Molecule(atom='H 0 0 0; H 0 0 1',basis_type='gto',basis='sto-3g')

    sto = Orbitals(
                basis_type=m.basis_type,
                nelec=m.nelec,
                atom_coords=m.atom_coords,
                norb = m.norb,
                nshells= m.nshells,
                index_ctr = m.index_ctr,
                bas_n=m.bas_n,
                bas_l=m.bas_l,
                bas_m=m.bas_m,
                bas_coeffs = m.bas_coeffs, 
                bas_exp=m.bas_exp,
                basis=m.basis)

    pos = torch.rand(20,sto.nelec*3)
    aoval = sto.forward(pos)