import torch
from torch import nn
import torch.nn.functional as F
from math import pi as PI

from pyCHAMP.wavefunction.spherical_harmonics import SphericalHarmonics

class STO_SZ(nn.Module):

    def __init__(self,
                nelec,
                atom_coords,
                nshells,
                bas_n,bas_l,bas_m, 
                bas_exp ):

        '''Radial Basis Function Layer in N dimension

        Args:
            nelec : number of electron
            atoms : position of the atoms
            nshells : number of bas per atom
            bas_n,bas_l,bas_m : quantum number of the bas
            bas_exp : exponent of the bas
        '''

        super(STO_SZ,self).__init__()

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

    def forward(self,input):
        
        # get the pos of the bas
        self.bas_coords = self.atom_coords.repeat_interleave(self.nshells,dim=0)
        print(self.bas_coords.shape)

        # get the x,y,z, distance component of each point from each RBF center
        # -> (Nbatch,Nelec,Nrbf,Ndim)
        xyz =  (input.view(-1,self.nelec,1,self.ndim) - self.bas_coords[None,...])
        print(xyz.shape)

        # compute the distance
        # -> (Nbatch,Nelec,Nrbf)
        R = torch.sqrt((xyz**2).sum(3))
        print(R.shape)

        # radial part
        # -> (Nbatch,Nelec,Nrbf)
        X = R**self.bas_n * torch.exp(-self.bas_exp*R)
        print(X.shape)

        # compute by the spherical harmonics
        # -> (Nbatch,Nelec,Nrbf)
        Y = SphericalHarmonics(xyz,self.bas_l,self.bas_m)
        print(Y.shape)

        return X * Y

if __name__ == "__main__":

    nelec = 2
    atoms = ['O','O']
    atom_coords = [[0.,0.,0.],[0.,0.,1.]]

    nshells = [5,5] #[1s,2s,2px,2py,2pz]
    bas_exp = [ 7.66, #O 1S
                2.25, #O 2S
                2.25,2.25,2.25, #O 2P
                7.66, #O 1S
                2.25, #O 2S
                2.25,2.25,2.25 #O 2P
                ]

    bas_n = [ 0, #1s
              1, #2s
              1,1,1, #2p
              0, #1s
              1, #2s             
              1,1,1 #2p 
              ]
  
    bas_l = [ 0, #1s
              0, #2s
              1,1,1, #2p
              0, #1s
              0, #2s             
              1,1,1 #2p 
              ]

    bas_m = [ 0, #1s
              0, #2s
              -1,0,1, #2p
              0, #1s
              0, #2s             
              -1,0,1 #2p 
              ]

    sto = STO_SZ(nelec=8,
                atom_coords=atom_coords,
                nshells=nshells,
                bas_n=bas_n,
                bas_l=bas_l,
                bas_m=bas_l, 
                bas_exp=bas_exp)

    pos = torch.rand(20,sto.nelec*3)
    aoval = sto.forward(pos)