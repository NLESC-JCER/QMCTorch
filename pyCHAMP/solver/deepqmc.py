import inspect

import numpy as np 

import torch
from torch import nn
from torch.autograd import Variable, grad
import torch.optim as optim
from torch.utils.data import DataLoader

from pyCHAMP.solver.solver_base import SOLVER_BASE
from pyCHAMP.solver.torch_utils import QMCDataSet, QMCLoss
from pyCHAMP.solver.refine_mesh import refine_mesh

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

    def modify_grid(self,centers,fc_weight):
        self.wf.rbf.centers.data = torch.tensor(centers)
        self.wf.rbf.sigma = self.wf.rbf.get_sigma(self.wf.rbf.centers) 
        self.wf.fc.weight.data = torch.tensor(fc_weight)

    def get_observable(self,obs_dict,pos,**kwargs):
        '''compute all the requuired observable.

        Args :
            obs_dict : a dictionanry with all keys 
                        corresponding to a method of self.wf
            **kwargs : the possible arguments for the methods
        TODO : match the signature of the callables
        '''

        for obs in obs_dict.keys():

            # get the method
            func = self.wf.__getattribute__(obs)
            data = func(pos).detach().numpy()
            obs_dict[obs].append(data)

        return obs_dict

    def train(self,nepoch,
              batchsize=32,
              pos=None,
              obs_dict=None,
              ntherm=-1,
              resample=100,
              resample_every=25,
              loss='variance',
              plot = None,
              refine = None):
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
            obs_dict = {'local_energy':[]}

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
            
                if self.wf.fc.clip:
                    self.wf.fc.apply(clipper)
                
            if plot is not None:
                plot.drawNow()

            obs_dict = self.get_observable(obs_dict,pos)
            print('epoch %d loss %f' %(n,cumulative_loss))
            print('variance : %f' %np.var(obs_dict['local_energy'][-1]))
            print('energy : %f' %np.mean(obs_dict['local_energy'][-1]) )
            print('distance : %f' %self.wf.atomic_distance() )
            print('sigma : %f' %self.wf.get_sigma() )
            
            
            if n%resample_every == 0:
                #pos = self.sample(pos=pos.detach().numpy(),ntherm=ntherm,with_tqdm=False)
                pos = self.sample(pos=None,ntherm=ntherm,with_tqdm=True)
                self.dataloader.dataset.data = pos

        return pos, obs_dict



