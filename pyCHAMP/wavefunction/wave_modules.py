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

        out = torch.zeros(input.shape[0],self.nconfs)
        for isample in range(input.shape[0]):

            for ic,(cup,cdown) in enumerate(zip(self.configs[0],self.configs[1])):

                mo_up = input[isample].index_select(0,self.index_up).index_select(1,cup)
                mo_down = input[isample].index_select(0,self.index_down).index_select(1,cdown)
                out[isample,ic] = torch.det(mo_up) * torch.det(mo_down)

        return out

class ElectronDistance(torch.autograd.Function):
        
    @staticmethod
    def forward(self,input):
        batch_size = input.shape[0]
        inp_norm = (input**2).sum(2).view(batch_size,-1,1)
        dist = inp_norm + inp_norm.view(batch_size,1,-1) - 2.0 * torch.bmm(input,input.transpose(1,2))
        return dist #view(batch_size,-1,1)

class TwoBodyJastrowFactor(nn.Module):

    def __init__(self,nup,ndown):
        super(TwoBodyJastrowFactor, self).__init__()

        self.nup = nup
        self.ndown = ndown

        self.weight = Variable(torch.tensor([1.0]))
        self.weight.requires_grad = True
        bup = torch.cat( (0.25*torch.ones(nup,nup),0.5*torch.ones(nup,ndown) ),dim=1)
        bdown = torch.cat( (0.5*torch.ones(ndown,nup),0.25*torch.ones(ndown,ndown) ),dim=1)
        self.static_weight = torch.cat( (bup,bdown),dim=0)

    def forward(self,input):
        return JastrowFunction.apply(input, self.weight, self.static_weight)

class JastrowFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx,input,weight,static_weight):
        ctx.save_for_backward(input, weight, static_weight)
        output = static_weight * input / (1.0 + weight * input)
        return output #.sum([1,2])

    @staticmethod
    def backward(ctx,grad_output):
        input, weight, static_weight = ctx.saved_tensors
        grad_input = (static_weight / (1+weight*input) *( 1 -  input * weight / (1+weight*input) ) )
        grad_weight = -(static_weight * input**2 * (1+weight*input)**(-2) )
        return grad_output * grad_input, grad_output * grad_weight, None

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