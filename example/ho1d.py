import torch
from torch.optim import Adam

from deepqmc.sampler.metropolis import  Metropolis
from deepqmc.wavefunction.wf_potential import Potential
from deepqmc.solver.solver_potential import SolverPotential 
from deepqmc.solver.plot import plot_results_1d, plotter1d

def pot_func(pos):
    '''Potential function desired.'''
    return  0.5*pos**2

def ho1d_sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    return torch.exp(-0.5*pos**2)

# box
domain, ncenter = {'xmin':-5.,'xmax':5.}, 5

# wavefunction
wf = Potential(pot_func,domain,ncenter,nelec=1)

#sampler
sampler = Metropolis(nwalkers=250, nstep=1000, 
                     step_size = 1., nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-5,'max':5})

# optimizer
opt = Adam(wf.parameters(),lr=0.01)

# define solver
solver = SolverPotential(wf=wf,sampler=sampler,optimizer=opt)

# train the wave function
pos,obs_dict = solver.run(100, loss = 'variance',
                         plot = plotter1d(wf,domain,50,sol=ho1d_sol) )

# plot the final wave function 
plot_results_1d(solver,obs_dict,domain,50,ho1d_sol,e0=0.5)







