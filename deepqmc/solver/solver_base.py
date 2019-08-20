import numpy as np
import torch
from types import SimpleNamespace


class SolverBase(object):

    def __init__(self,wf=None, sampler=None, optimizer=None):

        self.wf = wf
        self.sampler = sampler
        self.opt = optimizer  


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

    def sample(self,ntherm=-1,with_tqdm=True,pos=None):
        ''' sample the wave function.'''
        
        pos = self.sampler.generate(self.wf.pdf,ntherm=ntherm,with_tqdm=with_tqdm,pos=pos)
        pos.requires_grad = True
        return pos.float()

    def get_observable(self,obs_dict,pos,**kwargs):
        '''compute all the required observable.

        Args :
            obs_dict : a dictionanry with all keys 
                        corresponding to a method of self.wf
            **kwargs : the possible arguments for the methods
        TODO : match the signature of the callables
        '''

        for obs in self. obs_dict.keys():

            # get the method
            func = self.wf.__getattribute__(obs)
            data = func(pos)
            if isinstance(data,torch.Tensor):
                data = data.detach().numpy()
            self.obs_dict[obs].append(data)

    def get_wf(self,x):
        '''Get the value of the wave functions at x.'''
        vals = self.wf(x)
        return vals.detach().numpy().flatten()

    def energy(self,pos=None):
        '''Get the energy of the wave function.'''
        if pos is None:
            pos = self.sample(ntherm=-1)
        return self.wf.energy(pos)

    def variance(self,pos):
        '''Get the variance of the wave function.'''
        if pos is None:
            pos = self.sample(ntherm=-1)
        return self.wf.variance(pos)

    def single_point(self,pos=None,prt=True):
        '''Performs a single point calculation.'''
        if pos is None:
            pos = self.sample(ntherm=-1)
        e = self.energy(pos)
        s = self.variance(pos)
        if prt:
            print('Energy   : ',e)
            print('Variance : ',s)
        return pos, e, s

    def save_checkpoint(self,epoch,loss,filename):
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : self.wf.state_dict(),
            'optimzier_state_dict' : self.opt.state_dict(),
            'loss' : loss
            }, filename)
        return loss

    def run(self):
        raise NotImplementedError()
