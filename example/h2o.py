import torch
from torch import optim
from torch.optim import Adam

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital


from deepqmc.sampler.walkers import Walkers
from deepqmc.sampler.metropolis import Metropolis
from deepqmc.wavefunction.molecule import Molecule

from deepqmc.solver.plot_data import plot_observable

# define the molecule
mol = Molecule(atom='water.xyz', unit='angs',
               basis_type='gto', basis='sto-3g')

# define the wave function
wf = Orbital(mol, kinetic_jacobi=True,
             configs='ground_state', use_projector=True)

# sampler
sampler = Metropolis(nwalkers=100, nstep=500, step_size=0.25,
                     nelec=wf.nelec, ndim=wf.ndim, init=mol.domain('normal'))

# optimizer
opt = Adam(wf.parameters(), lr=0.1)

# scheduler
scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.75)

# solver
solver = SolverOrbital(wf=wf, sampler=sampler,
                       optimizer=opt, scheduler=scheduler)

# solver.configure(task='wf_opt')
# pos, e, v = solver.single_point(ntherm=-1, ndecor=10)

pos = solver.sample(ntherm=0, ndecor=10)
obs = solver.sampling_traj(pos)
plot_observable(obs, e0=-74, ax=None)

# # optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')
# solver.save_traj('h2o_traj.xyz')

# # plot the data
# plot_observable(solver.obs_dict)
