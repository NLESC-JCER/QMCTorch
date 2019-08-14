import torch
from torch import nn
import torch.nn.functional as F
from math import pi as PI
import numpy as np

from deepqmc.wavefunction.spherical_harmonics import SphericalHarmonics

class AtomicOrbitals(nn.Module):

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

        super(AtomicOrbitals,self).__init__()

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

        # get the cartesian exponents
        #self.lmn_cart = self.get_lmn_cart()

        # select the radial aprt
        radial_dict = {'sto':self._radial_slater, 
                       'gto':self._radial_gaussian,
                       'gto_cart':self._radial_gausian_cart}
        self.radial = radial_dict[basis_type]

        # get the normaliationconstants
        self.norm_cst = self.get_norm(basis_type)

    def get_norm(self,basis_type):

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
        '''
        nfact = torch.tensor([np.math.factorial(2*n) for n in self.bas_n],dtype=torch.float)
        return (2*self.bas_exp)**self.bas_n * torch.sqrt(2*self.bas_exp / nfact)

    def _norm_gaussian(self):
        '''Normalisation of the gto
        phi = N * r**n * exp(-alpha*r**2)
        see http://fisica.ciens.ucv.ve/~svincenz/TISPISGIMR.pdf 3.326 2.10 page 337
        '''
        beta = 2*self.bas_exp
        nfact = torch.tensor([np.math.factorial(n) for n in self.bas_n],dtype=torch.float)
        twonfact = torch.tensor([np.math.factorial(2*n) for n in self.bas_n],dtype=torch.float)
        return torch.sqrt(2 * nfact / twonfact * ( 4*beta )**self.bas_n * torch.sqrt(beta/np.pi))

    def _norm_gaussian_cart(self):
        '''Normlization of cartesian gaussian functions
        taken from http://www.chem.unifr.ch/cd/lectures/files/module5.pdf
        '''
        from scipy.special import factorial2 as f2
        L = self.lmn_cart.sum(1)
        num = 2**L * self.bas_coeffs**((2*L+3)/4)
        denom = torch.sqrt(f2(2*self.lmn-1).prod(1))
        return (2./np.pi)**(3./4.)  * num / denom

    def _radial_slater(self,R):
        return R**self.bas_n * torch.exp(-self.bas_exp*R)

    def _radial_gaussian(self,R):
        return R**self.bas_n * torch.exp(-self.bas_exp*R**2)

    def _radial_gausian_cart(self,xyz,R):
        raise NotImplementedError('Cartesian GTOs are on the to do list')
        #return xyz**self.lmn_cart * torch.exp(-self.bas_exp*R**2)

    def forward(self,input):
        
        nbatch = input.shape[0]

        # get the pos of the bas
        self.bas_coords = self.atom_coords.repeat_interleave(self.nshells,dim=0)

        # get the x,y,z, distance component of each point from each RBF center
        # -> (Nbatch,Nelec,Nbas,Ndim)
        xyz =  (input.view(-1,self.nelec,1,self.ndim) - self.bas_coords[None,...])
        
        # compute the distance
        # -> (Nbatch,Nelec,Nbas)
        R = torch.sqrt((xyz**2).sum(3))
        
        # radial part
        # -> (Nbatch,Nelec,Nbas)
        R = self.radial(R)
        
        # compute by the spherical harmonics
        # -> (Nbatch,Nelec,Nbas)
        Y = SphericalHarmonics(xyz,self.bas_l,self.bas_m)
        
        # product with coefficients
        # -> (Nbatch,Nelec,Nbas)
        bas = self.norm_cst * self.bas_coeffs * R * Y

        # contract the basis
        # -> (Nbatch,Nelec,Norb)
        ao = torch.zeros(nbatch,self.nelec,self.norb)
        ao.index_add_(2,self.index_ctr,bas)

        return ao




if __name__ == "__main__":

    from pyCHAMP.wavefunction.molecule import Molecule

    #m = Molecule(atom='H 0 0 0; H 0 0 1',basis_type='sto',basis='dz')
    m = Molecule(atom='H 0 0 0; H 0 0 1',basis_type='gto',basis='sto-3g')

    sto = AtomicOrbitals(
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