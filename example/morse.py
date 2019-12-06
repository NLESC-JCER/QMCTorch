import torch
from torch import optim

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.wavefunction.wf_potential import Potential
from deepqmc.solver.solver_potential import SolverPotential
from deepqmc.solver.plot_potential import plot_results_1d, plotter1d


def pot_func(pos):
    '''Potential function desired.'''
    return 0.5*(torch.exp(-2.*(pos)) - 2.*torch.exp(-pos)).view(-1, 1)


def ho1d_sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    vn = torch.exp(-torch.exp(-pos)-0.5*pos)
    return vn / torch.max(vn)


# box
domain, ncenter = {'xmin': -3., 'xmax': 8.}, 11

# wavefunction
wf = Potential(pot_func, domain, ncenter, fcinit='random', nelec=1, sigma=0.5)

# sampler
sampler = Metropolis(nwalkers=1000, nstep=2000,
                     step_size=1., nelec=wf.nelec,
                     ndim=wf.ndim, domain={'min': -5, 'max': 5})

# optimizer
opt = optim.Adam(wf.parameters(), lr=0.05)
# opt = optim.SGD(wf.parameters(),lr=0.05)

scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.75)

# define solver
solver = SolverPotential(wf=wf, sampler=sampler,
                         optimizer=opt, scheduler=scheduler)

# train the wave function
plotter = plotter1d(wf, domain, 100, sol=ho1d_sol)
solver.run(300, loss='variance', plot=plotter, save='model.pth')

# plot the final wave function
plot_results_1d(solver, domain, 100, ho1d_sol, e0=-0.125, load='model.pth')
