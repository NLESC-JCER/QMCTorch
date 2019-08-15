import torch
import torch.optim as optim

from deepqmc.wavefunction.wf_potential import Potential
from deepqmc.sampler.metropolis import  Metropolis
from deepqmc.solver.deepqmc import DeepQMC

from deepqmc.solver.plot import plot_results_1d as plot_results
from deepqmc.solver.plot import plotter1d, plot_wf_1d

import matplotlib.pyplot as plt

import numpy as np


def pot_func(pos):
    '''Potential function desired.'''
    return  0.5*pos**2

def ho1d_sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    vn = torch.exp(-0.5*pos**2)
    return vn 

# box
domain, ncenter = {'xmin':-5.,'xmax':5.}, 5

# wavefunction
wf = Potential(pot_func,domain,ncenter,nelec=1)

#sampler
sampler = Metropolis(nwalkers=500, nstep=1000, 
                     step_size = 1., nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-5,'max':5})

# optimizer
opt = optim.Adam(wf.parameters(),lr=0.01)

# define solver
qmc = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)

# single point calculation
pos, e, v = qmc.single_point()
print('energy   :', e, '\nvariance :', v)

plot_wf_1d(qmc,domain,20,grad=False,hist=False,pot=False,sol=ho1d_sol,ax=None)

pos,obs_dict = qmc.train(100,
         batchsize=500,
         pos = pos,
         resample=100,
         resample_from_last = True,
         resample_every = 1,
         ntherm = -1,
         loss = 'variance',
         plot = plotter1d(wf,domain,50,sol=ho1d_sol) )

plot_results(qmc,obs_dict,domain,50,ho1d_sol,e0=0.5)







