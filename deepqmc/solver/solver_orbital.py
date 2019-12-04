import os
import time
from tqdm import tqdm
import numpy as np 

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from mendeleev import element

from deepqmc.solver.solver_base import SolverBase
from deepqmc.solver.torch_utils import DataSet, Loss, ZeroOneClipper, OrthoReg


def printd(rank,*args):
    if rank == 1:
        print(*args)


class SolverOrbital(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None,
                 scheduler=None):
        SolverBase.__init__(self,wf,sampler,optimizer)
        self.scheduler = scheduler

        # task
        self.configure(task='geo_opt')

        #esampling
        self.resampling(ntherm=-1, resample=100,resample_from_last=True, resample_every=1)

        # observalbe
        self.observable(['local_energy'])

        # distributed model
        self.save_model = 'model.pth'


    def configure(self,task='wf_opt',freeze=[]):
        '''Configure the optimzier for specific tasks.'''
        self.task = task

        if task == 'geo_opt':
            self.wf.ao.atom_coords.requires_grad = True
            self.wf.ao.bas_exp.requires_grad = False
            for param in self.wf.mo.parameters():
                param.requires_grad = False
            self.wf.fc.weight.requires_grad = False
            

        elif task == 'wf_opt':
            self.wf.ao.bas_exp.requires_grad = True
            for param in self.wf.mo.parameters():
                param.requires_grad = True
            self.wf.fc.weight.requires_grad = True
            self.wf.ao.atom_coords.requires_grad = False  

            for name in freeze:
                if name.lower() == 'ci':
                    self.wf.fc.weight.requires_grad = False
                elif name.lower() == 'mo':
                    for param in self.wf.mo.parameters():
                        param.requires_grad = False
                elif name.lower() == 'bas_exp':
                    self.wf.ao.bas_exp.requires_grad = False
                else:
                    opt_freeze = ['ci','mo','bas_exp']
                    raise ValueError('Valid arguments for freeze are :', opt_freeze)

    def run(self, nepoch, batchsize=None, loss='variance'):

        '''Train the model.

        Arg:
            nepoch : number of epoch
            batchsize : size of the minibatch, if None take all points at once
            loss : loss used ('energy','variance' or callable (for supervised)
        '''

        #sample the wave function
        pos = self.sample(ntherm=self.resample.ntherm)

        # handle the batch size        
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

        # orthogonalization penalty for the MO coeffs
        self.ortho_loss = OrthoReg()
                
        # clipper for the fc weights
        clipper = ZeroOneClipper()
    
        cumulative_loss = []
        min_loss = 1E3

        # get the initial observalbe

        self.get_observable(self.obs_dict,pos)
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

                # compute local gradients
                self.opt.zero_grad()
                loss.backward()

                # agregate gradient
                # if necesary
                self.wf.sum_mo_grad()

                # optimize
                self.opt.step()

                if self.wf.fc.clip:
                    self.wf.fc.apply(clipper)

            if cumulative_loss < min_loss:
                min_loss = self.save_checkpoint(n,cumulative_loss,self.save_model)
                 

            self.get_observable(self.obs_dict,pos)
            print('loss %f' %(cumulative_loss))
            for k in self.obs_dict:
                if k == 'local_energy':
                    print('variance : %f' %np.var(self.obs_dict['local_energy'][-1]))
                    print('energy : %f' %np.mean(self.obs_dict['local_energy'][-1]) )
                else:
                    print(k + ' : ', self.obs_dict[k][-1])

            print('----------------------------------------')
            
            # resample the data
            if (n%self.resample.resample_every == 0) or (n == nepoch-1):
                if self.resample.resample_from_last:
                    pos = pos.clone().detach()
                else:
                    pos = None
                pos = self.sample(pos=pos,ntherm=self.resample.ntherm,with_tqdm=False)
                
                self.dataloader.dataset.data = pos

            if self.task=='geo_opt':
                self.wf.update_mo_coeffs()

            if self.scheduler is not None:
                self.scheduler.step()

        #restore the sampler number of step
        self.sampler.nstep = _nstep_save

    def save_traj(self,fname):

        f = open(fname,'w')
        xyz = self.obs_dict['geometry']
        natom = len(xyz[0])
        conv2bohr = 1.88973
        for snap in xyz:
            f.write('%d \n\n' %natom)
            for at in snap:
                f.write('%s % 7.5f % 7.5f %7.5f\n' %(at[0],at[1][0]/conv2bohr,
                                                           at[1][1]/conv2bohr,
                                                           at[1][2]/conv2bohr))
            f.write('\n')
        f.close()


    



