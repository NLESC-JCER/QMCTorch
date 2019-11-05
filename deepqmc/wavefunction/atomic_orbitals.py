import torch
from torch import nn
import torch.nn.functional as F
from math import pi as PI
import numpy as np

from deepqmc.wavefunction.spherical_harmonics import SphericalHarmonics

class AtomicOrbitals(nn.Module):

    def __init__(self,mol):

        '''Radial Basis Function Layer in N dimension

        Args:
            mol: the molecule 
        '''

        super(AtomicOrbitals,self).__init__()

        # wavefunction data
        self.nelec = mol.nelec
        self.norb = mol.norb
        self.ndim = 3

        # make the atomic position optmizable
        self.atom_coords = nn.Parameter(torch.tensor(mol.atom_coords))
        self.atom_coords.requires_grad = True
        self.natoms = len(self.atom_coords)
        self.atomic_number = mol.atomic_number

        # define the BAS positions
        self.nshells = torch.tensor(mol.nshells)
        self.bas_coords = self.atom_coords.repeat_interleave(self.nshells,dim=0)
        self.nbas = len(self.bas_coords)

        #index for the contractions
        self.index_ctr = torch.tensor(mol.index_ctr)

        # get the coeffs of the bas
        self.bas_coeffs = torch.tensor(mol.bas_coeffs)

        # get the exponents of the bas
        self.bas_exp = nn.Parameter(torch.tensor(mol.bas_exp))
        self.bas_exp.requires_grad = True

        # get the quantum number
        self.bas_n = torch.tensor(mol.bas_n).float()
        self.bas_l = torch.tensor(mol.bas_l)
        self.bas_m = torch.tensor(mol.bas_m)

        # select the radial aprt
        radial_dict = {'sto':self._radial_slater, 
                       'gto':self._radial_gaussian,
                       'gto_cart':self._radial_gausian_cart}

        self.radial = radial_dict[mol.basis_type]
        
        # get the normaliationconstants
        self.norm_cst = self.get_norm(mol.basis_type)

    def get_norm(self,basis_type):

        with torch.no_grad():

            #select the normalization
            if basis_type == 'sto':
                return self._norm_slater()
            elif basis_type == 'gto':
                return self._norm_gaussian()
            elif basis_type == 'gto_cart':
                return self._norm_gaussian_cart()


    def _norm_slater(self):
        '''Normalization of the STO 
        taken from www.theochem.ru.nl/~pwormer/Knowino/knowino.org/wiki/Slater_orbital.html

        C Filippi Multiconf wave functions for QMC of first row diatomic molecules
        JCP 105, 213 1996

        '''
        nfact = torch.tensor([np.math.factorial(2*n) for n in self.bas_n],dtype=torch.float32)
        return (2*self.bas_exp)**self.bas_n * torch.sqrt(2*self.bas_exp / nfact)

    def _norm_gaussian(self):
        '''Computational Quantum Chemistry: An interactive Intrduction to basis set theory
            eq: 1.14 page 23.'''

        from scipy.special import factorial2 as f2

        bas_n = self.bas_n+1.
        exp1 = 0.25*(2.*bas_n+1.)

        A = self.bas_exp**exp1
        B = 2**(2.*bas_n+3./2)
        C = torch.tensor(f2(2*bas_n.int()-1)*np.pi**0.5).float()
        
        return torch.sqrt(B/C)*A

    def _radial_slater(self, R, xyz=None, derivative=0):
        if derivative == 0:
            return R**self.bas_n * torch.exp(-self.bas_exp*R)

        elif derivative > 0:
            sum_xyz = xyz.sum(3)
            rn = R**(self.bas_n)
            nabla_rn = self.bas_n * sum_xyz * R**(self.bas_n-2)
            er = torch.exp(-self.bas_exp*R)
            nabla_er = - self.bas_exp * sum_xyz * er 

            if derivative == 1:
                return nabla_rn*er + rn*nabla_er 


    def _radial_gaussian(self,R, xyz=None, derivative=0):
        if derivative == 0:
            return R**self.bas_n * torch.exp(-self.bas_exp*R**2)
            

        elif derivative > 0:
            
            sum_xyz = xyz.sum(3)
            
            rn = R**(self.bas_n)
            nabla_rn = self.bas_n * sum_xyz * R**(self.bas_n-2)

            er = torch.exp(-self.bas_exp*R**2)
            nabla_er = -2*self.bas_exp * sum_xyz * er

            if derivative == 1:
                return nabla_rn *er + rn*nabla_er 
                

            elif derivative == 2:

                lap_rn = self.bas_n * ( 3*R**(self.bas_n-2) \
                    + (xyz**2).sum(3) * (self.bas_n-2) * R**(self.bas_n-4) )

                lap_er = 4 * self.bas_exp**2 * (xyz**2).sum(3) * er \
                - 6 * self.bas_exp * er
                
                print((lap_rn*er).shape)
                print((nabla_rn*nabla_er).shape)
                print((rn*lap_er).shape)
                return lap_rn*er + 2*nabla_rn*nabla_er + rn*lap_er
                

    def _radial_gausian_cart(self,R,xyz=None, derivative=0):
        raise NotImplementedError('Cartesian GTOs are on the to do list')
        #return xyz**self.lmn_cart * torch.exp(-self.bas_exp*R**2)


    def forward(self,input,derivative=0):
        
        nbatch = input.shape[0]

        # get the pos of the bas
        self.bas_coords = self.atom_coords.repeat_interleave(self.nshells,dim=0)

        # get the x,y,z, distance component of each point from each RBF center
        # -> (Nbatch,Nelec,Nbas,Ndim)
        xyz =  (input.view(-1,self.nelec,1,self.ndim) - self.bas_coords[None,...])
        
        # compute the distance
        # -> (Nbatch,Nelec,Nbas)
        r = torch.sqrt((xyz**2).sum(3))
        
        # radial part
        # -> (Nbatch,Nelec,Nbas)
        R = self.radial(r)
        
        # compute by the spherical harmonics
        # -> (Nbatch,Nelec,Nbas)
        Y = SphericalHarmonics(xyz,self.bas_l,self.bas_m)

        # treat the different cases
        # of derivative values
        if derivative == 0 :
            #bas = R * Y
            bas = R

        elif derivative == 1 :
            dR = self.radial(r,xyz=xyz,derivative=1)
            dY = SphericalHarmonics(xyz,self.bas_l,self.bas_m,derivative=1)
            #bas = dR * Y  + R * dY
            bas = dR

        elif derivative == 2:            
            dR = self.radial(r,xyz=xyz,derivative=1)
            dY = SphericalHarmonics(xyz,self.bas_l,self.bas_m,derivative=1)
            
            d2R = self.radial(r,xyz=xyz,derivative=2)
            d2Y = SphericalHarmonics(xyz,self.bas_l,self.bas_m,derivative=2)

            #bas = d2R * Y + 2. * dR * dY + R * d2Y
            bas = d2R
        
        # product with coefficients
        # -> (Nbatch,Nelec,Nbas)
        bas = self.norm_cst * self.bas_coeffs * bas

        # contract the basis
        # -> (Nbatch,Nelec,Norb)
        ao = torch.zeros(nbatch,self.nelec,self.norb)
        ao.index_add_(2,self.index_ctr,bas)

        return ao

if __name__ == "__main__":

    from deepqmc.wavefunction.molecule import Molecule

    #m = Molecule(atom='H 0 0 0; H 0 0 1',basis_type='sto',basis='dz')
    m = Molecule(atom='Li 0 0 0; H 0 0 3.015',basis_type='gto',basis='sto-3g')

    ao = AtomicOrbitals(m)

    pos = torch.rand(20,ao.nelec*3)
    aoval = ao.forward(pos,derivative=1)