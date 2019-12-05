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
from torch.multiprocessing import Process, Queue, Event, Manager
from mendeleev import element

from deepqmc.solver.solver_orbital import SolverOrbital
from deepqmc.solver.torch_utils import DataSet, Loss, ZeroOneClipper, OrthoReg


def printd(rank,*args):
    if rank == 1:
        print(*args)

class DistSolverOrbital(SolverOrbital):

    def __init__(self, wf=None, sampler=None, optimizer=None):
        SolverOrbital.__init__(self,wf,sampler,optimizer)

        # task
        self.configure(task='geo_opt')

        #esampling
        self.resampling(ntherm=-1, resample=100,resample_from_last=True, resample_every=1)

        # observalbe
        self.observable(['local_energy'])

        # distributed model
        self.conf_dist(master_address='127.0.0.1',master_port='29500',backend='gloo')

        self.save_model = 'model.pth'


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

            manager = Manager()
            obs_data = manager.list()

            for rank in range(ndist):
                p = Process(target=self.init_process,
                            args=( obs_data, rank, ndist, nepoch, batchsize, loss ))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        self.obs_dict = obs_data

    def init_process(self, obs_data, rank, size, nepoch, batchsize, loss):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = self.master_address
        os.environ['MASTER_PORT'] = self.master_port
        dist.init_process_group(self.dist_backend, rank=rank, world_size=size)
        self._worker(nepoch,batchsize,loss)
        obs_data.append(self.obs_dict['local_energy'])

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

        # gather all the data on all procs
        #self.gather_obs_dict()

    def average_gradients(self):
        '''Average the gradients of all the distributed processes.'''
        size = float(dist.get_world_size())
        for param in self.wf.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data,op=dist.ReduceOp.SUM)
                param.grad.data /= size


    def gather_obs_dict(self):
        for k in self.obs_dict.keys():
            data = self.obs_dict[k]
            data_gather = [torch.zeros_like(data)] * dist.get_world_size()
            dist.all_gather(data_gather,data)
            self.obs_dict[k] = data_gather[data]
