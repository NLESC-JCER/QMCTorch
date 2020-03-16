import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, SGD, lr_scheduler

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital
from deepqmc.solver.torch_utils import set_torch_double_precision
from deepqmc.sampler.metropolis import Metropolis
#from deepqmc.sampler.metropolis_update_ao import Metropolis
#from deepqmc.sampler.metropolis_kalos import Metropolis
from deepqmc.optim.sr import StochasticReconfiguration

from deepqmc.wavefunction.molecule import Molecule
from deepqmc.solver.plot_data import plot_observable, load_observable, save_observalbe, plot_block, plot_walkers_traj

# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree

set_torch_double_precision()


# define the molecule
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
               basis_type='sto',
               basis='dzp',
               unit='bohr')


# define the wave function
wf = Orbital(mol, kinetic='jacobi',
             configs='ground_state',
             use_jastrow=True)

wf.jastrow.weight.data[0] = 1

# sampler
sampler = Metropolis(nwalkers=250, nstep=5000, step_size=0.2,
                     ndim=wf.ndim, nelec=wf.nelec,
                     init=mol.domain('atomic'),
                     move={'type': 'all-elec', 'proba': 'normal'}, wf=wf)
# wf=wf)

# optimizer
lr_dict = [{'params': wf.jastrow.parameters(), 'lr': 1E-2},
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

#pos, e, v = solver.single_point(ntherm=1000, ndecor=200)
# pos = solver.sample(ntherm=0, ndecor=0)
# obs = solver.sampling_traj(pos)
# Tc = plot_walkers_traj(obs)
# plot_block(obs)

# save_observalbe('obs.pkl', obs)
# obs = load_observable('obs.pkl')
# plot_observable(obs, e0=-1.1645, ax=None, var=True)


# optimize the wave function
if 1:
    solver.configure(task='wf_opt', freeze=['ao', 'mo'])
    solver.observable(['local_energy'])
    solver.initial_sampling(ntherm=1000, ndecor=200)
    solver.resampling(nstep=100, step_size=0.2, resample_every=1, tqdm=True)
    solver.ortho_mo = False
    data = solver.run(5, batchsize=None,
                      loss='energy',
                      grad='manual',
                      clip_loss=False)
    save_observalbe('h2.pkl', solver.obs_dict)
    plot_observable(solver.obs_dict, e0=-1.1645)

# # optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')

# plot the data
# plot_observable(solver.obs_dict, e0=-1.16)
