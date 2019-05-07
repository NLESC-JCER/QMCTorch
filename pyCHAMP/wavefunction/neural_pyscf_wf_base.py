import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from pyscf import scf, gto, mcscf




class WaveNet(nn.Module):

    def __init__(self,atom,basis,active_space):

        super(WaveNet, self).__init__()

        self.atom = atom
        self.basis = basis
        self.active_space = active_space

        self.ndim = 3

        # molecule
        self.mol = gto.M(atom=self.atom,basis=self.basis)
        self.nelec = np.sum(self.mol.nelec)

        # mean field
        self.rhf = scf.RHF(self.mol).run()
        self.nmo, self.nao = self.rhf.mo_coeff.shape

        # multi configuration
        #self.mc = mcscf.CASSCF(self.rhf,active_space[0],active_space[1])
        #self.nci = self.mc.ci

        # get the configs
        self.configs, self.ci_coeffs, self.nci = self.select_configuration()
        
        # transofrm the AO in MO
        self.layer_mo = nn.Linear(self.nao,self.nmo,bias=False)
        self.layer_mo.weight = nn.Parameter(torch.tensor(self.rhf.mo_coeff))

        # determinant pooling
        self.sdpool = SlaterPooling(self.configs)

        # CI Layer
        self.layer_ci = nn.Linear(self.nci,1,bias=False)
        self.layer_ci.weight = nn.Parameter(torch.tensor(self.ci_coeffs))

    def forward(self, x):

        x = self.ao_values(x)
        x = self.layer_mo(x)
        x = self.sdpool(x)
        x = self.layer_ci(x)
        
        return x


    def ao_values(self,pos,func='GTOval_sph'):
        '''Returns the values of all atomic orbitals
        on the position specified by pos.

        Args :
            pos : ndarray shape(N,3)
            func : str Method name
        Returns:
            values : nd array shape(N,Nbasis)
        '''
        return self.mol.eval_gto(func,pos)


    def select_configuration(self):
        
        
        confs = []
        coeffs = []

        c0 = np.argwhere(self.rhf.mo_occ!=0).flatten()
        index_homo = np.max(c0)

        confs.append(c0)
        coeffs.append(1.)
        nci = 1

        nocc = int(np.ceil(self.active_space[0]/2))
        nvirt = int(self.active_space[0]/2)
        for iocc in range(nocc):
            for ivirt in range(1,nvirt+1):
                c = list(np.copy(c0))
                print(c)
                c.pop(index_homo-iocc) 
                c.append(index_homo+ivirt)
                confs.append(np.array(c))
                coeffs.append(0)
                nci += 1

        return torch.LongTensor(confs), coeffs, nci



class SlaterPooling(nn.Module):

    """Applies a slater determinant pooling in the active space."""

    def __init__(self,configs):
        super(SlaterPooling, self).__init__()
        self.configs = configs
        self.nconfs = len(configs)


    def forward(self,input):
        out = torch.zeros(input.shape[0],self.nconfs)
        for isample in range(input.shape[0]):
            for ic in range(self.nconfs):
                c = self.configs[ic]
                mo = input[isample].index_select(0,c).index_select(1,c)
                out[isample,ic] = torch.det(mo)
        return out


if __name__ == "__main__" :

    wf = WaveNet(atom='O 0 0 0; H 0 1 0; H 0 0 1',basis='dzp',active_space=(2,2))