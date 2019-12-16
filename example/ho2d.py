from torch import optim

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.wavefunction.wf_potential_2d import Potential2D as Potential
from deepqmc.solver.solver_potential import SolverPotential
from deepqmc.solver.plot_potential import plot_results_2d, plotter2d, plot_wf_2d


def pot_func(pos):
    '''Potential function desired.'''
    return 0.5*(pos**2).sum(-1)


# box
domain, ncenter = {'min': -5., 'max': 5.}, 21

# wavefunction
wf = Potential(pot_func, domain, ncenter, ndim=2,
               fcinit='random', sigma=0.5)

# sampler
sampler = Metropolis(nwalkers=1000, nstep=2000,
                     step_size=1., nelec=wf.nelec,
                     ndim=wf.ndim, init={'min': -5, 'max': 5})

# optimizer
opt = optim.Adam(wf.parameters(), lr=0.05)
# opt = optim.SGD(wf.parameters(),lr=0.05)

scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.75)

# define solver
solver = SolverPotential(wf=wf, sampler=sampler,
                         optimizer=opt, scheduler=scheduler)

# plot_wf_2d(solver, domain, [11, 11])
# pos, e, v = solver.single_point()

# train the wave function
plotter = plotter2d(wf, domain, [51, 51], save='./image/')
solver.run(250, loss='variance', plot=plotter, save='model.pth')

# # plot the final wave function
plot_results_2d(solver, domain, [51, 51], e0=1., load='model.pth')
