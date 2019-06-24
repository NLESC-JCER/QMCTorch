import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

import numpy as np
from pyscf import scf, gto, mcscf

from tqdm import tqdm
from time import time

from pyCHAMP.wavefunction.wave_modules import (SlaterPooling,
                                               ElectronDistance,
                                               TwoBodyJastrowFactor,
                                               AOLayer)

class NEURAL_PYSCF_WF(nn.Module):

    def __init__(self,atom,basis,active_space):

        super(NEURAL_PYSCF_WF, self).__init__()

        self.atom = atom
        self.basis = basis
        self.active_space = active_space

        self.ndim = 3

        # molecule
        self.mol = gto.M(atom=self.atom,basis=self.basis)
        self.nup, self.ndown = self.mol.nelec
        self.nelec = np.sum(self.mol.nelec)

        # mean field
        self.rhf = scf.RHF(self.mol).run()
        self.nmo, self.nao = self.rhf.mo_coeff.shape

        # multi configuration
        #self.mc = mcscf.CASSCF(self.rhf,active_space[0],active_space[1])
        #self.nci = self.mc.ci

        # get the configs
        self.configs = self.select_configuration_singlet()
        self.nci = len(self.configs[0])

        self.ci_coeffs = np.zeros(self.nci)
        self.ci_coeffs[0] = 1

        # transform pos in AO
        self.layer_ao = AOLayer(self.mol)

        # transofrm the AO in MO
        self.layer_mo = nn.Linear(self.nao,self.nmo,bias=False)
        self.mo_coeff = self.normalize_cols(self.rhf.mo_coeff)
        self.layer_mo.weight = nn.Parameter(Variable(torch.tensor(self.mo_coeff).transpose(0,1)))

        # determinant pooling
        self.sdpool = SlaterPooling(self.configs,self.nup,self.ndown)

        # CI Layer
        self.layer_ci = nn.Linear(self.nci,1,bias=False)
        self.layer_ci.weight = nn.Parameter(torch.tensor(self.ci_coeffs).float())

    def forward(self, x):
        
        batch_size = x.shape[0]
        x = x.view(batch_size,-1,3)

        #t0 = time()
        #x = AOFunction.apply(x,self.mol)
        x = self.layer_ao(x)
        #print(x)
        #print(" __ __ AO : %f" %(time()-t0))

        #t0 = time()
        x = self.layer_mo(x)
        #print(x)
        #print(" __ __ MO : %f" %(time()-t0))

        #t0 = time()
        x = self.sdpool(x)
        #print(x)
        #print(" __ __ SD : %f" %(time()-t0))

        #t0 = time()
        x = self.layer_ci(x)
        #print(x)
        #print(" __ __ CI : %f" %(time()-t0))
        
        return x

    @staticmethod
    def normalize_cols(mat):
        n,m = mat.shape
        for i in range(m):
            mat[:,i] = mat[:,i] / np.linalg.norm(mat[:,i])
        return mat

    def ao_values(self,pos,func='GTOval_sph'):
        '''Returns the values of all atomic orbitals
        on the position specified by pos.

        Args :
            pos : ndarray shape(N,3)
            func : str Method name
        Returns:
            values : nd array shape(N,Nbasis)
        '''
        return torch.tensor([self.mol.eval_gto(func,p) for p in  pos])


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
                c.pop(index_homo-iocc) 
                c.append(index_homo+ivirt)
                confs.append(np.array(c))
                coeffs.append(0)
                nci += 1

        return torch.LongTensor(confs), coeffs, nci


    def select_configuration_singlet(self):
        
        confs_spin_up = []
        confs_spin_down = []
        
        c0 = np.argwhere(self.rhf.mo_occ!=0).flatten()
        index_homo = np.max(c0)

        confs_spin_up.append(c0)
        confs_spin_down.append(c0)

        nocc = int(np.ceil(self.active_space[0]/2))
        nvirt = int(self.active_space[0]/2)

        for iocc in range(nocc):

            for ivirt in range(1,nvirt+1):

                cup = list(np.copy(c0))

                cup.pop(index_homo-iocc) 
                cup.append(index_homo+ivirt)

                confs_spin_up.append(np.array(cup))
                confs_spin_down.append(np.array(c0))

        return torch.LongTensor(confs_spin_up), torch.LongTensor(confs_spin_down)


    def nuclear_potential(self,pos):
        
        nwalker = pos.shape[0]
        pot = torch.zeros(nwalker)

        for ielec in range(self.nelec):
            epos = pos[:,ielec*self.ndim:(ielec+1)*self.ndim]
            for atpos,atmass in zip(self.mol.atom_coords(),self.mol.atom_mass_list()):
                r = torch.sqrt( ((epos-torch.tensor(atpos).float())**2).sum(1) ).float()
                pot -= (atmass / r)

        return pot

    def electronic_potential(self,pos):

        nwalker = pos.shape[0]
        pot = torch.zeros(nwalker)
        

        for ielec1 in range(self.nelec-1):
            epos1 = pos[:,ielec1*self.ndim:(ielec1+1)*self.ndim]
            
            for ielec2 in range(ielec1+1,self.nelec):
                epos2 = pos[:,ielec2*self.ndim:(ielec2+1)*self.ndim]
                
                r = torch.sqrt( ((epos1-epos2)**2).sum(1) ).float()
                pot -= (1./r)

        return pot

    def nuclear_repulsion(self):
        '''Compute the nuclear repulsion of the system.'''

        coords = self.mol.atom_coords()
        mass = self.mol.atom_mass_list()
        natom = len(mass)

        vnn = 0.
        for i1 in range(natom-1):
            c1 = torch.tensor(coords[i1,:])
            m1 = torch.tensor(mass[i1])
            for i2 in range(i1+1,natom):
                c2 = torch.tensor(coords[i2,:])
                m2 = torch.tensor(mass[i2])
                r = torch.sqrt( ((c1-c2)**2).sum() ).float()
                vnn += m1*m2 / r
        return vnn

    def pdf(self,pos):
        return self.forward(pos)**2

    # def kinetic_autograd(self,pos):

    #     out = self.forward(pos)
    #     z = Variable(torch.ones(out.shape))
    #     jacob = grad(out,pos,grad_outputs=z,create_graph=True)[0]
    #     hess = grad(jacob.sum(),pos,create_graph=True)[0]
    #     return hess.sum(1)

    def kinetic_energy(self,pos,out=None):
        '''Compute the second derivative of the network
        output w.r.t the value of the input. 

        This is to compute the value of the kinetic operator.

        Args:
            pos: position of the electron
            out : preomputed values of the wf at pos

        Returns:
            values of nabla^2 * Psi
        '''

        if out is None:
            out = self.forward(pos)

        # compute the jacobian            
        z = Variable(torch.ones(out.shape))
        jacob = grad(out,pos,grad_outputs=z,create_graph=True)[0]

        # compute the diagonal element of the Hessian
        z = Variable(torch.ones(jacob.shape[0]))
        hess = torch.zeros(jacob.shape[0])
        for idim in range(jacob.shape[1]):
            tmp = grad(jacob[:,idim],pos,grad_outputs=z,create_graph=True,allow_unused=True)[0]    
            hess += tmp[:,idim]

        return -0.5 * hess.view(-1,1)
    
    def kinetic_energy_finite_difference(self,pos,eps=1E-6):
        '''Compute the second derivative of the network
        output w.r.t the value of the input using finite difference.

        This is to compute the value of the kinetic operator.

        Args:
            pos: position of the electron
            out : preomputed values of the wf at pos
            eps : psilon for numerical derivative
        Returns:
            values of nabla^2 * Psi
        '''

        nwalk = pos.shape[0]
        ndim = pos.shape[1]
        out = torch.zeros(nwalk,1)

        for icol in range(ndim):

            pos_tmp = pos.clone()
            feps = -2*self.forward(pos_tmp)

            pos_tmp = pos.clone()
            pos_tmp[:,icol] += eps
            feps += self.forward(pos_tmp)

            pos_tmp = pos.clone()
            pos_tmp[:,icol] -= eps
            feps += self.forward(pos_tmp)

            out += feps/(eps**2)

        return -0.5*out.view(-1,1)
    
    def local_energy(self,pos):
        ''' local energy of the sampling points.'''
        return self.kinetic_energy(pos)/self.forward(pos) \
               + self.nuclear_potential(pos)  \
               + self.electronic_potential(pos) \
               + self.nuclear_repulsion()

    def energy(self,pos):
        '''Total energy for the sampling points.'''
        return torch.mean(self.local_energy(pos))

    def variance(self, pos):
        '''Variance of the energy at the sampling points.'''
        return torch.var(self.local_energy(pos))




if __name__ == "__main__" :

    from torch.autograd import gradcheck
    #wf = NEURAL_PYSCF_WF(atom='O 0 0 0; H 0 1 0; H 0 0 1',basis='dzp',active_space=(1,1))
    wf = NEURAL_PYSCF_WF(atom='H 0 0 0; H 0 0 1',basis='dzp',active_space=(1,1))
    nwalkers = 25

    pos = -2 + 4*torch.rand(nwalkers,wf.ndim*wf.nelec).float()
    pos.requires_grad = True
    out = wf(pos)
    out.backward(torch.ones(out.shape).float())

    #inp = pos.view(nwalkers,-1,3)
    k = wf.kinetic_autograd(pos)


    x = pos.view(25,-1,3)
    D = ElectronDistance.apply(x)
    J2 = TwoBodyJastrowFactor(1,1)