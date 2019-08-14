import torch
import torch.optim as optim



from deepqmc.solver.deepqmc import DeepQMC
from deepqmc.sampler.metropolis import  METROPOLIS
from deepqmc.sampler.hamiltonian import HAMILTONIAN

from deepqmc.wavefunction.wf_pinbox import PinBox

from deepqmc.solver.plot import plot_results_1d as plot_results
from deepqmc.solver.plot import plotter1d, plot_wf_1d

import matplotlib.pyplot as plt

import numpy as np


def pot_func(pos):
    return  0.5*pos**2

def ho1d_sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    vn = torch.exp(-0.5*pos**2)
    return vn #/np.linalg.norm(vn)

# wavefunction
domain = {'xmin':-5,'xmax':5}
ncenter = [25]
wf = PinBox(pot_func,domain,ncenter,nelec=1)

#sampler
sampler = METROPOLIS(nwalkers=250, nstep=100, 
                     step_size = 3., nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-5,'max':5})

#sampler
sampler_ham = HAMILTONIAN(nwalkers=250, nstep=100, 
                     step_size = 0.01, nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-5,'max':5}, L=5)

# optimizer
opt = optim.Adam(wf.parameters(),lr=0.005)

# network
net = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)
pos = net.sample(ntherm=-1)
print('energy   :', net.wf.energy(pos))
print('variance :', net.wf.variance(pos))
# pos = None
# obs_dict = None

boundary = 5.
domain = {'xmin':-boundary,'xmax':boundary}
ncenter = 51

plot_wf_1d(net,domain,ncenter,sol=ho1d_sol,
           hist=False,
           pot=False,
           grad=True)

# plt.ion()
# domain = {'xmin':-5.,'xmax':5.}
# plot1D = plotter1d(wf,domain,50,sol=ho1d_sol)



    # net.wf.fc.weight.requires_grad = True
    # net.wf.rbf.centers.requires_grad = False

    # pos,obs_dict = net.train(250,
    #          batchsize=250,
    #          pos = pos,
    #          obs_dict = obs_dict,
    #          resample=100,
    #          ntherm=-1,
    #          loss = 'variance',
    #          plot=plot1D,
    #          refine=25)

    # net.wf.fc.weight.requires_grad = False
    # net.wf.rbf.centers.requires_grad = True

    # pos,obs_dict = net.train(10,
    #          batchsize=250,
    #          pos = pos,
    #          obs_dict = obs_dict,
    #          resample=100,
    #          ntherm=-1,
    #          loss = 'variance',
    #          sol=ho1d_sol,
    #          fig=fig)

#plot_results(net,obs_dict,ho1d_sol,e0=0.5)






