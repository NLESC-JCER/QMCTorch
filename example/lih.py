from torch import optim
from torch.optim import Adam, SGD

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital
from deepqmc.solver.torch_utils import set_torch_double_precision

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.wavefunction.molecule import Molecule

from deepqmc.solver.plot_data import plot_observable


set_torch_double_precision()

# define the molecule
mol = Molecule(atom='Li 0 0 0; H 0 0 3.015',
               basis_type='sto',
               basis='dz',
               unit='bohr')

# define the wave function
wf = Orbital(mol, kinetic='jacobi',
             configs='cas(2,2)',
             use_jastrow=True)

# sampler
sampler = Metropolis(nwalkers=2000, nstep=1000, step_size=0.01,
                     nelec=wf.nelec, ndim=wf.ndim,
                     init=mol.domain('center'),
                     move={'type': 'all-elec-iter', 'proba': 'normal'})

# optimizer
opt = Adam(wf.parameters(), lr=1E-1)
# opt = SGD(wf.parameters(), lr=1E-3, momentum=0.)

# scheduler
scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.75)

# solver
solver = SolverOrbital(wf=wf, sampler=sampler,
                       optimizer=opt, scheduler=None)

# pos, e, v = solver.single_point(ntherm=-1, ndecor=100)

# pos = solver.sample(ntherm=0, ndecor=10)
# obs = solver.sampling_traj(pos)
# plot_observable(obs, e0=-8., ax=None)

# optimize the wave function
solver.configure(task='wf_opt', freeze=['mo', 'bas_exp'])
solver.observable(['local_energy'])
solver.initial_sampling(ntherm=-1, ndecor=100)
solver.resampling(nstep=25, resample_every=1)
solver.sampler.step_size = 1E-3
data = solver.run(50, loss='energy', clip_loss=True)
plot_observable(solver.obs_dict, e0=-8.06)

# # optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')
# solver.save_traj('h2o_traj.xyz')

# # plot the data
# plot_observable(solver.obs_dict)
