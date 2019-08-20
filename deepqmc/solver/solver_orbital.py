import os
import time
from tqdm import tqdm
import numpy as np 
from types import SimpleNamespace

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.multiprocessing import Process
from mendeleev import element

from deepqmc.solver.solver_base import SolverBase
from deepqmc.solver.torch_utils import DataSet, Loss, ZeroOneClipper, OrthoReg


def printd(rank,*args):
    if rank == 1:
        print(*args)


class SolverOrbital(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None):
        SolverBase.__init__(self,wf,sampler,optimizer)

        # task
        self.configure(task='geo_opt')

        #esampling
        self.resampling(ntherm=-1, resample=100,resample_from_last=True, resample_every=1)

        # observalbe
        self.observable(['local_energy'])

        # distributed model
        self.conf_dist(master_address='127.0.0.1',master_port='29500',backend='gloo')

        self.save_model = 'model.pth'


    def configure(self,task='wf_opt',freeze=[]):
        '''Configure the optimzier for specific tasks.'''
        self.task = task

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

    def resampling(self,ntherm=-1, resample=100,resample_from_last=True, resample_every=1):
        '''Configure the resampling options.'''
        self.resample = SimpleNamespace()
        self.resample.ntherm = ntherm
        self.resample.resample = resample
        self.resample.resample_from_last = resample_from_last
        self.resample.resample_every = resample_every

    def observable(self,obs):
        '''Create the observalbe we want to track.'''
        self.obs_dict = {}
        for k in obs:
            self.obs_dict[k] = []
        if 'local_energy' not in self.obs_dict:
            self.obs_dict['local_energy'] = []

    def conf_dist(self,master_address='127.0.0.1',master_port='29500',backend='gloo'):
        '''Configure the communicatin address and backend.'''
        self.master_address = master_address
        self.master_port = master_port
        self.dist_backend = backend

    def run(self, nepoch, batchsize=None, loss='variance', ndist=1 ):

        if ndist == 1:
            self.distributed_training = False
            self._worker(nepoch,batchsize,loss)

        else:

            self.distributed_training = True
            processes = []
            for rank in range(ndist):
                p = Process(target=self.init_process,
                            args=( rank, ndist, nepoch, batchsize, loss  ))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()


    def init_process(self, rank, size, nepoch, batchsize, loss):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = self.master_address
        os.environ['MASTER_PORT'] = self.master_port
        dist.init_process_group(self.dist_backend, rank=rank, world_size=size)
        self._worker(nepoch,batchsize,loss)

    def _worker(self, nepoch, batchsize, loss ):

        '''Train the model.

        Arg:
            nepoch : number of epoch
            batchsize : size of the minibatch, if None take all points at once
            loss : loss used ('energy','variance' or callable (for supervised)
        '''

        # get the rank of the worker
        if self.distributed_training:
            rank = dist.get_rank()
        else:
            rank = 1

        # reconfigure the sampler if we have dist training
        if self.distributed_training:
            size = int(dist.get_world_size())
            self.sampler.nwalkers //= size
            self.sampler.walkers.nwalkers //= size

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

        for n in range(nepoch):
            printd(rank,'----------------------------------------')
            printd(rank,'epoch %d' %n)

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

                #average gradients
                if self.distributed_training :
                    self.average_gradients()

                # optimize
                self.opt.step()

                if self.wf.fc.clip:
                    self.wf.fc.apply(clipper)

            if cumulative_loss < min_loss:
                min_loss = self.save_checkpoint(n,cumulative_loss,self.save_model)
                 

            self.get_observable(self.obs_dict,pos)
            printd(rank,'loss %f' %(cumulative_loss))
            for k in self.obs_dict:
                if k =='local_energy':
                    printd(rank,'variance : %f' %np.var(self.obs_dict['local_energy'][-1]))
                    printd(rank,'energy : %f' %np.mean(self.obs_dict['local_energy'][-1]) )
                else:
                    printd(rank,k + ' : ', self.obs_dict[k][-1])

            printd(rank,'----------------------------------------')
            
            # resample the data
            if (n%self.resample.resample_every == 0) or (n == nepoch-1):
                if self.resample.resample_from_last:
                    pos = pos.clone().detach()
                else:
                    pos = None
                pos = self.sample(pos=pos,ntherm=self.resample.ntherm,with_tqdm=False)
                
                self.dataloader.dataset.data = pos

        #restore the sampler number of step
        self.sampler.nstep = _nstep_save

    def average_gradients(self):
        '''Average the gradients of all the distributed processes.'''
        size = float(dist.get_world_size())
        for param in self.wf.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data,op=dist.ReduceOp.SUM)
                param.grad.data /= size





    

