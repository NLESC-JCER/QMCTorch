import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

import numpy as np
from pyscf import scf, gto, mcscf

from tqdm import tqdm
from time import time


class SlaterPooling(nn.Module):

    """Applies a slater determinant pooling in the active space."""

    def __init__(self,configs,nup,ndown):
        super(SlaterPooling, self).__init__()

        self.configs = configs
        self.nconfs = len(configs[0])

        self.index_up = torch.arange(nup)
        self.index_down = torch.arange(nup,nup+ndown)

    def forward(self,input):

        ''' Compute the product of spin up/down determinants
        Args:
            input : MO values (Nbatch, Nelec, Nmo)
        Returnn:
            determiant (Nbatch, Ndet)
        '''
        nbatch = input.shape[0]
        out = torch.zeros(nbatch,self.nconfs)
        
        for ic,(cup,cdown) in enumerate(zip(self.configs[0],self.configs[1])):

            mo_up = input.index_select(1,self.index_up).index_select(2,cup)
            mo_down = input.index_select(1,self.index_down).index_select(2,cdown)

            # a batch version of det is on its way (end July 2019)
            # https://github.com/pytorch/pytorch/issues/7500
            # we'll move to that asap but in the mean time we loop
            for isample in range(nbatch):
                out[isample,ic] = torch.det(mo_up[isample]) * torch.det(mo_down[isample])

        return out


class ElectronDistance(nn.Module):
    
    def __init__(self,nelec,ndim):
        super(ElectronDistance,self).__init__()
        self.nelec = nelec
        self.ndim = ndim
        
    def forward(self,input):
        '''compute the pairwise distance between two sets of electrons.
        Args:
            input1 (Nbatch,Nelec1*Ndim) : position of the electrons
            input2 (Nbatch,Nelec2*Ndim) : position of the electrons if None -> input1
        Returns:
            mat (Nbatch,Nelec1,Nelec2) : pairwise distance between electrons
        '''

        input = input.view(-1,self.nelec,self.ndim)
        norm = (input**2).sum(-1).unsqueeze(-1)
        dist = norm + norm.transpose(1,2) -2.0 * torch.bmm(input,input.transpose(1,2))

        return dist

class TwoBodyJastrowFactor(nn.Module):

    def __init__(self,nup,ndown):
        super(TwoBodyJastrowFactor, self).__init__()

        self.nup = nup
        self.ndown = ndown
        self.nelec = nup+ndown

        self.weight = Variable(torch.tensor([1.0]))
        self.weight.requires_grad = True

        bup = torch.cat( (0.25*torch.ones(nup,nup),0.5*torch.ones(nup,ndown) ),dim=1)
        bdown = torch.cat( (0.5*torch.ones(ndown,nup),0.25*torch.ones(ndown,ndown) ),dim=1)
        self.static_weight = torch.cat( (bup,bdown),dim=0)

    def forward(self,input):
        
        factors = torch.exp(self.static_weight * input / (1.0 + self.weight * input))
        factors = factors[:,torch.tril(torch.ones(self.nelec,self.nelec))==0].prod(1)
        return factors.view(-1,1)

        #return JastrowFunction.apply(input,self.weight,self.static_weight)

        
class JastrowFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx,input,weight,static_weight):
        '''Compute the Jastrow factor.
        Args:
            input : Nbatch x Nelec x Nelec (elec distance)
            weight : Nelec, Nelec
            static weight : Float
        Returns:
            jastrow : Nbatch x 1
        '''

        # save the tensors
        ctx.save_for_backward(input,weight,static_weight)

        # all jastrow for all electron pairs
        factors = torch.exp(static_weight * input / (1.0 + weight * input))
        
        # product of the off diag terms 
        nr,nc = input.shape[1], input.shape[2]
        factors = factors[:,torch.tril(torch.ones(nr,nc))==0].prod(1)
        
        return factors.view(-1,1)


    # @staticmethod
    # def backward(ctx,grad_output):
    #     input, weight, static_weight = ctx.saved_tensors
    #     grad_input = (static_weight / (1+weight*input) *( 1 -  input * weight / (1+weight*input) ) )
    #     grad_weight = -(static_weight * input**2 * (1+weight*input)**(-2) )
    #     return grad_output * grad_input, grad_output * grad_weight, None

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

        out = torch.zeros(input.shape)
        for k in range(3):
            out[:,:,k] = (grad_output * deriv_ao[:,k,:,:]).sum(-1)

        return out, None

class AOLayer(nn.Module):

    def __init__(self,mol):
        super(AOLayer,self).__init__()
        self.mol = mol

    def forward(self,input):
        return AOFunction.apply(input,self.mol)


if __name__ == "__main__":

    pos = torch.rand(10,12)
    edist = ElectronDistance.apply(pos)
    jastrow = TwoBodyJastrowFactor(2,2)
    val = jastrow(edist)