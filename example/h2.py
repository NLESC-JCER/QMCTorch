import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, SGD, lr_scheduler

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital
from deepqmc.utils.torch_utils import set_torch_double_precision
from deepqmc.sampler.metropolis import Metropolis

from deepqmc.wavefunction.molecule import Molecule
from deepqmc.utils.plot_data import (load_observable,
                                     save_observalbe, plot_block,
                                     plot_walkers_traj,
                                     plot_energy, plot_data)

# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree

set_torch_double_precision()


# define the molecule
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
               basis='dz',
               unit='bohr')


# define the wave function
wf = Orbital(mol, kinetic='jacobi',
             configs='cas(2,2)',
             use_jastrow=True)

wf.jastrow.weight.data[0] = 1.

# sampler
sampler = Metropolis(nwalkers=500, nstep=2000, step_size=0.2,
                     ndim=wf.ndim, nelec=wf.nelec,
                     init=mol.domain('atomic'),
                     move={'type': 'all-elec', 'proba': 'normal'}, wf=wf)
# wf=wf)

# optimizer
lr_dict = [{'params': wf.jastrow.parameters(), 'lr': 1E-3},
           {'params': wf.ao.parameters(), 'lr': 1E-6},
           {'params': wf.mo.parameters(), 'lr': 1E-3},
           {'params': wf.fc.parameters(), 'lr': 1E-3}]


opt = Adam(lr_dict, lr=1E-3)
# opt = SGD(lr_dict, lr=1E-1)
# opt = StochasticReconfiguration(wf.parameters(), wf)

# scheduler
scheduler = lr_scheduler.StepLR(opt, step_size=100, gamma=0.90)

# solver
solver = SolverOrbital(wf=wf, sampler=sampler,
                       optimizer=opt, scheduler=None)

if 1:
    pos, e, v = solver.single_point(ntherm=1000, ndecor=100)
    # pos = solver.sample(ntherm=1000, ndecor=100)
    # obs = solver.sampling_traj(pos)
    # Tc = plot_walkers_traj(obs)
    # plot_block(obs)

    # save_observalbe('obs.pkl', obs)
    # obs = load_observable('obs.pkl')
    # plot_energy(obs, e0=-1.1645, show_variance=True)


# optimize the wave function
if 0:
    solver.configure(task='wf_opt', freeze=['ao', 'mo'])
    solver.observable(['local_energy'])
    solver.initial_sampling(ntherm=1000, ndecor=100)

    solver.resampling(nstep=20, ntherm=-1, step_size=0.2,
                      resample_from_last=True,
                      resample_every=1, tqdm=True)

    solver.ortho_mo = False
    data = solver.run(50, batchsize=None,
                      loss='energy',
                      grad='manual',
                      clip_loss=False)

    save_observalbe('h2.pkl', solver.obs_dict)
    e, v = plot_energy(solver.obs_dict, e0=-1.1645, show_variance=True)
    plot_data(solver.obs_dict, obs='jastrow.weight')

# # optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')

# plot the data
# plot_observable(solver.obs_dict, e0=-1.16)
