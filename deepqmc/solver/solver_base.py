import torch
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np


class SolverBase(object):

    def __init__(self, wf=None, sampler=None, optimizer=None):

        self.wf = wf
        self.sampler = sampler
        self.opt = optimizer
        self.cuda = False
        self.device = torch.device('cpu')

    def resampling(self, ntherm=-1, nstep=100, resample_from_last=True,
                   resample_every=1):
        '''Configure the resampling options.'''
        self.resample = SimpleNamespace()
        self.resample.ntherm = ntherm
        self.resample.resample = nstep
        self.resample.resample_from_last = resample_from_last
        self.resample.resample_every = resample_every

    def initial_sampling(self, ntherm=-1, ndecor=100):
        '''Configure the initial sampling options.'''
        self.initial_sample = SimpleNamespace()
        self.initial_sample.ntherm = ntherm
        self.initial_sample.ndecor = ndecor

    def observable(self, obs):
        '''Create the observalbe we want to track.'''
        self.obs_dict = {}

        for k in obs:
            self.obs_dict[k] = []

        if 'local_energy' not in self.obs_dict:
            self.obs_dict['local_energy'] = []

        if self.task == 'geo_opt' and 'geometry' not in self.obs_dict:
            self.obs_dict['geometry'] = []

    def sample(self, ntherm=-1, ndecor=100, with_tqdm=True, pos=None):
        ''' sample the wave function.'''

        pos = self.sampler.generate(
            self.wf.pdf, ntherm=ntherm, ndecor=ndecor,
            with_tqdm=with_tqdm, pos=pos)
        pos.requires_grad = True
        return pos

    def get_observable(self, obs_dict, pos, local_energy=None, **kwargs):
        '''compute all the required observable.

        Args :
            obs_dict : a dictionanry with all keys
                        corresponding to a method of self.wf
            **kwargs : the possible arguments for the methods
        TODO : match the signature of the callables
        '''

        if self.wf.cuda and pos.device.type == 'cpu':
            pos = pos.to(self.device)

        for obs in self. obs_dict.keys():

            if obs == 'local_energy' and local_energy is not None:
                data = local_energy.cpu().detach().numpy()
            else:
                # get the method
                func = self.wf.__getattribute__(obs)
                data = func(pos)
                if isinstance(data, torch.Tensor):
                    data = data.cpu().detach().numpy()

            self.obs_dict[obs].append(data)

    def print_observable(self, cumulative_loss):

        print('loss %f' % (cumulative_loss))
        for k in self.obs_dict:
            if k == 'local_energy':
                print('variance : %f' %
                      np.var(self.obs_dict['local_energy'][-1]))
                print('energy : %f' %
                      np.mean(self.obs_dict['local_energy'][-1]))
            else:
                print(k + ' : ', self.obs_dict[k][-1])

    def get_wf(self, x):
        '''Get the value of the wave functions at x.'''
        vals = self.wf(x)
        return vals.detach().numpy().flatten()

    def energy(self, pos=None):
        '''Get the energy of the wave function.'''
        if pos is None:
            pos = self.sample(ntherm=-1)

        if self.wf.cuda and pos.device.type == 'cpu':
            pos = pos.to(self.device)

        return self.wf.energy(pos)

    def variance(self, pos):
        '''Get the variance of the wave function.'''
        if pos is None:
            pos = self.sample(ntherm=-1)

        if self.wf.cuda and pos.device.type == 'cpu':
            pos = pos.to(self.device)

        return self.wf.variance(pos)

    def single_point(self, pos=None, prt=True,
                     with_tqdm=True, ntherm=-1, ndecor=100):
        '''Performs a single point calculation.'''
        if pos is None:
            pos = self.sample(ntherm=ntherm, ndecor=ndecor,
                              with_tqdm=with_tqdm)

        if self.wf.cuda and pos.device.type == 'cpu':
            pos = pos.to(self.device)

        e, s = self.wf._energy_variance(pos)
        if prt:
            print('Energy   : ', e)
            print('Variance : ', s)
        return pos, e, s

    def save_checkpoint(self, epoch, loss, filename):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.wf.state_dict(),
            'optimzier_state_dict': self.opt.state_dict(),
            'loss': loss
        }, filename)
        return loss

    def sampling_traj(self, pos):
        ndim = pos.shape[-1]
        p = pos.view(-1, self.sampler.nwalkers, ndim)
        el = []
        for ip in tqdm(p):
            el.append(self.wf.local_energy(ip).detach().numpy())
        return {'local_energy': el, 'pos': p}

    def run(self, nepoch, batchsize=None, loss='variance'):
        raise NotImplementedError()
