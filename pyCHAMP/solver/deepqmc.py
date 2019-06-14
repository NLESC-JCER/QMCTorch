import numpy as np 

import torch
from torch import nn
from torch.autograd import Variable, grad
import torch.optim as optim
from torch.utils.data import DataLoader

from pyCHAMP.solver.solver_base import SOLVER_BASE
from pyCHAMP.solver.torch_utils import QMCDataSet, QMCLoss

import matplotlib.pyplot as plt

from tqdm import tqdm
import time


class UnitNormClipper(object):

    def __call__(self,module):
        if hasattr(module,'weight'):
            w = module.weight.data
            w.div_(torch.norm(w).expand_as(w))

class ZeroOneClipper(object):
    
    def __call__(self, module):
        if hasattr(module,'weight'):
            w = module.weight.data
            w.sub_(torch.min(w)).div_(torch.norm(w).expand_as(w))
            
class DeepQMC(SOLVER_BASE):

    def __init__(self, wf=None, sampler=None, optimizer=None):
        SOLVER_BASE.__init__(self,wf,sampler,None)
        self.opt = optimizer

    def sample(self,ntherm=-1,with_tqdm=True,pos=None):

        t0 = time.time()
        pos = self.sampler.generate(self.wf.pdf,ntherm=ntherm,with_tqdm=with_tqdm,pos=pos)
        pos = torch.tensor(pos)
        pos = pos.view(-1,self.sampler.ndim*self.sampler.nelec)
        pos.requires_grad = True
        return pos.float()

    def observalbe(self,func,pos):
        obs = []
        for p in tqdm(pos):
            obs.append( func(p).data.numpy().tolist() )
        return obs

    def get_wf(self,x):
        vals = self.wf(x)
        return vals.detach().numpy().flatten()

    def train(self,nepoch,
              batchsize=32,
              pos=None,
              obs_dict=None,
              ntherm=-1,
              resample=100,
              loss='variance',
              plot = None):
        '''Train the model.

        Arg:
            nepoch : number of epoch
            pos : presampled electronic poition
            ntherm : thermalization of the MC sampling
            nresample : number of MC step during the resampling
            loss : loss used ('energy','variance' or callable (for supervised)
            sol : anayltical solution for plotting (callable)
        '''

        if obs_dict is None:
            obs_dict = {'energy':[],'variance':[],'loss':[],'local_energy':[]}

        if pos is None:
            pos = self.sample(ntherm=ntherm)

        self.sampler.nstep=resample

        self.dataset = QMCDataSet(pos)
        self.dataloader = DataLoader(self.dataset,batch_size=batchsize)
        self.loss = QMCLoss(self.wf,method=loss)
                
        cumulative_loss = []
        clipper = ZeroOneClipper()
    
        for n in range(nepoch):

            cumulative_loss = 0
            for data in self.dataloader:
                
                lpos = Variable(data).float()
                lpos.requires_grad = True
                vals = self.wf(lpos)

                loss = self.loss(vals,lpos)
                cumulative_loss += loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.wf.fc.apply(clipper)

            if plot is not None:
                plot.drawNow()

            locale_ = self.wf.local_energy(pos)
            e_ = locale_.mean()
            v_ = locale_.var()

            print('epoch %d loss %f' %(n,cumulative_loss))
            print('variance : %f' %v_)
            print('energy : %f' %e_)

            obs_dict['energy'].append(e_.detach().numpy().tolist())
            obs_dict['variance'].append(v_.detach().numpy().tolist())
            obs_dict['local_energy'].append(locale_.detach().numpy().tolist())
            obs_dict['loss'].append(cumulative_loss.detach().numpy().tolist())

            if self.sampler.nstep > 0:
                pos = self.sample(pos=pos.detach().numpy(),ntherm=ntherm,with_tqdm=False)
                self.dataloader.dataset.data = pos


        return pos, obs_dict



