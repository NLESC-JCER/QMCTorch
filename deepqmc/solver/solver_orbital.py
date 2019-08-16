import time
import numpy as np 

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from mendeleev import element

from deepqmc.solver.solver_base import SolverBase
from deepqmc.solver.torch_utils import DataSet, Loss, ZeroOneClipper, OrthoReg

from tqdm import tqdm

from mayavi import mlab

class SolverOrbital(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None):
        SolverBase.__init__(self,wf,sampler,optimizer)

    def configure(self,task='wf_opt',freeze=[]):
        
        if task == 'geo_opt':
            self.wf.ao.bas_exp.requires_grad = False
            self.wf.mo.weight.requires_grad = False
            self.wf.fc.weight.requires_grad = False
            self.wf.ao.atom_coords.requires_grad = True

        elif task == 'wf_opt':
            self.wf.ao.bas_exp.requires_grad = True
            self.wf.mo.weight.requires_grad = True
            self.wf.fc.weight.requires_grad = True
            self.wf.ao.atom_coords.requires_grad = False  

            for name in freeze:
                if name.lower() == 'ci':
                    self.wf.fc.weight.requires_grad = False
                elif name.lower() == 'mo':
                    self.wf.mo.weight.requires_grad = False
                elif name.lower() == 'bas_exp':
                    self.wf.ao.bas_exp.requires_grad = False
                else:
                    opt_freeze = ['ci','mo','bas_exp']
                    raise ValueError('Valid arguments for freeze are :', opt_freeze)

    def run(self, nepoch, batchsize=None, pos=None, obs_dict=None, 
              ntherm=-1, resample=100, resample_from_last=True, resample_every=1,
              loss='variance', plot = None,
              save_model='model.pth'):

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

        if obs_dict is None:
            obs_dict = {'local_energy':[]}
        if 'local_energy' not in obs_dict:
            obs_dict['local_energy'] = []

        if pos is None:
            pos = self.sample(ntherm=ntherm)

        if batchsize is None:
            batchsize = len(pos)

        # change the number of steps
        _nstep_save = self.sampler.nstep
        self.sampler.nstep = resample

        # create the data loader
        self.dataset = DataSet(pos)
        self.dataloader = DataLoader(self.dataset,batch_size=batchsize)

        # get the loss
        self.loss = Loss(self.wf,method=loss)

        # orthogonalization penalty for the MO coeffs
        self.ortho_loss = OrthoReg()
                
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
                if self.wf.mo.weight.requires_grad:
                    loss += self.ortho_loss(self.wf.mo.weight)
                cumulative_loss += loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if self.wf.fc.clip:
                    self.wf.fc.apply(clipper)
                
            if plot is not None:
                plot.drawNow()

            if cumulative_loss < min_loss:
                min_loss = self.save_checkpoint(n,cumulative_loss,save_model)
                 

            obs_dict = self.get_observable(obs_dict,pos)
            print('loss %f' %(cumulative_loss))
            for k in obs_dict:
                if k =='local_energy':
                    print('variance : %f' %np.var(obs_dict['local_energy'][-1]))
                    print('energy : %f' %np.mean(obs_dict['local_energy'][-1]) )
                else:
                    print(k + ' : ', obs_dict[k][-1])

            print('----------------------------------------')
            
            
            
            
            if (n%resample_every == 0) or (n == nepoch-1):
                if resample_from_last:
                    pos = self.sample(pos=pos.detach().numpy(),ntherm=ntherm,with_tqdm=False)
                else:
                    pos = self.sample(pos=None,ntherm=ntherm,with_tqdm=False)
                self.dataloader.dataset.data = pos

        #restore the sampler number of step
        self.sampler.nstep = _nstep_save

        return pos, obs_dict






    

