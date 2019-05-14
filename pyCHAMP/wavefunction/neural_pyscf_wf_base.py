import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

import numpy as np
from pyscf import scf, gto, mcscf

from tqdm import tqdm
from time import time

class NEURAL_PYSCF_WF(nn.Module):

    def __init__(self,atom,basis,active_space):

        super(NEURAL_PYSCF_WF, self).__init__()

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
        self.configs = self.select_configuration_singlet()

        
        # transform pos in AO
        self.layer_ao = AOLayer(self.mol)

        # transofrm the AO in MO
        self.layer_mo = nn.Linear(self.nao,self.nmo,bias=False)
        self.layer_mo.weight = nn.Parameter(Variable(torch.tensor(self.rhf.mo_coeff).transpose(0,1)))

        # determinant pooling
        self.sdpool = SlaterPooling(self.configs)

        # CI Layer
        self.layer_ci = nn.Linear(self.nci,1,bias=False)
        self.layer_ci.weight = nn.Parameter(torch.tensor(self.ci_coeffs))

    def forward(self, x):
        
        batch_size = x.shape[0]
        x = x.view(batch_size,-1,3)

        #t0 = time()
        #x = self.ao_values(x)
        x = AOFunction.apply(x,self.mol)
        #x = self.layer_ao(x)
        #print(" __ __ AO : %f" %(time()-t0))

        #t0 = time()
        x = self.layer_mo(x)
        #print(" __ __ MO : %f" %(time()-t0))

        #t0 = time()
        x = self.sdpool(x)
        #print(" __ __ SD : %f" %(time()-t0))

        #t0 = time()
        x = self.layer_ci(x)
        #print(" __ __ CI : %f" %(time()-t0))
        
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

        ground_state =  list(self.rhf.mo_occ)
        configs = []
        configs.append(ground_state)

        nocc = int(np.ceil(self.active_space[0]/2))
        nvirt = int(self.active_space[0]/2)

        for iocc in range(nocc):
            for ivirt in range(nvirt):
                c = ground_state.copy()
                c[iocc] -= 1
                c[ivirt] += 1
                configs.append(c)

        return configs

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

    def pdf(self,pos):
        return self.forward(pos)**2

    def kinetic_fd(self,pos,eps=1E-6):

        '''Compute the action of the kinetic operator on the we points.
        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * psi
        '''

        nwalk = pos.shape[0]
        ndim = pos.shape[1]
        out = torch.zeros(nwalk)
        for icol in (range(ndim)):

            pos_tmp = pos.clone()
            feps = -2*self.forward(pos_tmp)            

            pos_tmp = pos.clone()
            pos_tmp[:,icol] += eps
            feps += self.forward(pos_tmp)
            
            pos_tmp = pos.clone()
            pos_tmp[:,icol] -= eps
            feps += self.forward(pos_tmp)

            out += feps/(eps**2)

        return out

    def kinetic_autograd(self,pos):

        out = self.forward(pos)
        z = Variable(torch.ones(out.shape))
        jacob = grad(out,pos,grad_outputs=z,create_graph=True)[0]
        hess = grad(jacob.sum(),pos,create_graph=True)[0]
        return hess.sum(1)


    def applyK(self,pos):
        '''Comute the result of H * psi

        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * pis
        ''' 
        Kpsi = -0.5*self.kinetic_autograd(pos) 
        return Kpsi

    
    def local_energy(self,pos):
        ''' local energy of the sampling points.'''
        return self.applyK(pos)/self.forward(pos) \
               + self.nuclear_potential(pos)  \
               + self.electronic_potential(pos)

    def energy(self,pos):
        '''Total energy for the sampling points.'''
        return torch.mean(self.local_energy(pos))

    def variance(self, pos):
        '''Variance of the energy at the sampling points.'''
        return torch.var(self.local_energy(pos))

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
                #mo = input[isample].index_select(0,c).index_select(1,c)


                print(c)
                print(input[isample])

                mo = self.get_mo(input[isample],c)               
                print(mo)

                out[isample,ic] = torch.det(mo)
        return out

    @staticmethod
    def duplicate_orbitals(mo):
        nelec,norb_ = mo.shape
        norb = 2*norb_
        return torch.stack( (mo,mo),dim=0).view(nelec,norb).t().contiguous().view(nelec,norb)

    @staticmethod
    def get_index_configs(config):
        norb = len(config)
        index = []
        for i in range(norb):
            if config[i] == 2:
                index.append(2*i)
                index.append(2*i+1)
            elif config[i] == 1:
                index.append(2*i)
        return torch.LongTensor(index)

    
    def get_mo(self,mo,conf):
        all_mos = self.duplicate_orbitals(mo)
        index = self.get_index_configs(conf)
        return all_mos.index_select(0,index).index_select(1,index)



class AOFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mol):
        ctx.save_for_backward(input)
        ctx.mol = mol
        pos = input.detach().numpy().astype('float64') 
        output = [mol.eval_gto("GTOval_sph",p) for p in pos]
        return torch.tensor(output,requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        pos = input.detach().numpy().astype('float64')
        deriv_ao = torch.tensor([ctx.mol.eval_gto("GTOval_ip_sph",p) for p in pos])
        print('GRAD OUT\n', grad_output)
        print('DERIV AO\n', deriv_ao.shape)

        out = torch.zeros(input.shape)
        for k in range(3):
            out[:,:,k] = (grad_output * deriv_ao[:,k,:,:]).sum(-1)
        print(out)
        return out, None

class AOLayer(nn.Module):

    def __init__(self,mol):
        super(AOLayer,self).__init__()
        self.mol = mol

    def forward(self,input):
        return AOFunction.apply(input,self.mol)

class AOLayer(nn.Module):

    def __init__(self,mol):
        super(AOLayer,self).__init__()
        self.mol = mol

    def forward(self,input):
        return AOFunction.apply(input,self.mol)


if __name__ == "__main__" :

    #wf = NEURAL_PYSCF_WF(atom='O 0 0 0; H 0 1 0; H 0 0 1',basis='dzp',active_space=(4,4))
    wf = NEURAL_PYSCF_WF(atom='H 0 0 0; H 0 0 1',basis='dzp',active_space=(1,1))
    nwalkers = 10
    pos = torch.rand(nwalkers,wf.ndim*wf.nelec).float()
    pos.requires_grad = True
    out = wf(pos)
    #out.backward(torch.ones(out.shape).float())
    inp = pos.view(nwalkers,-1,3)

    #k = wf.kinetic_autograd(inp)
