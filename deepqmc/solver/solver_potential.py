import inspect
import numpy as np 

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from deepqmc.solver.solver_base import SolverBase
from deepqmc.solver.torch_utils import DataSet, Loss, ZeroOneClipper


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from tqdm import tqdm
import time

class SolverPotential(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None,
                 scheduler=None):
        SolverBase.__init__(self,wf,sampler,optimizer)
        self.scheduler = scheduler

        #esampling
        self.resampling(ntherm=-1, resample=100,resample_from_last=True, resample_every=1)

        # observalbe
        self.observable(['local_energy'])

    def run(self,nepoch, batchsize=None, save='model.pth',  loss='variance', plot = None):

        '''Train the model.

        Arg:
            nepoch : number of epoch
            batchsize : size of the minibatch, if None take all points at once
            pos : presampled electronic poition
            obs_dict (dict, {name: []} ) : quantities to be computed during the training
                                           'name' must refer to a method of the Solver instance
            ntherm : thermalization of the MC sampling. If negative (-N) takes the last N entries
            resample : number of MC step during the resampling
            resample_from_last (bool) : if true use the previous position as starting for the resampling
            resample_every (int) : number of epch between resampling
            loss : loss used ('energy','variance' or callable (for supervised)
            plot : None or plotter instance from plot_utils.py to interactively monitor the training
        '''

        # checkpoint file
        self.save_model = save

        # sample the wave function
        pos = self.sample(ntherm=self.resample.ntherm)

        # determine the batching mode
        if batchsize is None:
            batchsize = len(pos)

        # change the number of steps
        _nstep_save = self.sampler.nstep
        self.sampler.nstep = self.resample.resample

        # create the data loader
        self.dataset = DataSet(pos)
        self.dataloader = DataLoader(self.dataset,batch_size=batchsize)

        # get the loss
        self.loss = Loss(self.wf,method=loss)
                
        # clipper for the fc weights
        clipper = ZeroOneClipper()
    
        cumulative_loss = []
        min_loss = 1E3

        for n in range(nepoch):
            print('----------------------------------------')
            print('epoch %d' %n)

            cumulative_loss = 0
            for data in self.dataloader:
                
                lpos = Variable(data).float()
                lpos.requires_grad = True

                loss = self.loss(lpos)
                cumulative_loss += loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if self.wf.fc.clip:
                    self.wf.fc.apply(clipper)
                
            if plot is not None:
                plot.drawNow()

            if cumulative_loss < min_loss:
                min_loss = self.save_checkpoint(n,cumulative_loss,self.save_model)
                 
            # get the observalbes
            self.get_observable(self.obs_dict,pos)
            print('loss %f' %(cumulative_loss))
            print('variance : %f' %np.var(self.obs_dict['local_energy'][-1]))
            print('energy : %f' %np.mean(self.obs_dict['local_energy'][-1]) )   
            print('----------------------------------------')
            
            # resample the data
            if (n%self.resample.resample_every == 0) or (n == nepoch-1):
                if self.resample.resample_from_last:
                    pos = pos.clone().detach()
                else:
                    pos = None
                pos = self.sample(pos=pos,ntherm=self.resample.ntherm,with_tqdm=False)
                self.dataloader.dataset.data = pos

            if self.scheduler is not None:
                self.scheduler.step()

        #restore the sampler number of step
        self.sampler.nstep = _nstep_save




