from torch import optim
from torch.optim import Adam

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital
from deepqmc.solver.torch_utils import set_torch_double_precision

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.wavefunction.molecule import Molecule

from deepqmc.solver.plot_data import plot_observable


set_torch_double_precision()

# define the molecule
mol = Molecule(atom='Li 0 0 0; H 0 0 3.015',
               basis_type='gto',
               basis='sto-6g',
               unit='bohr')

# define the wave function
wf = Orbital(mol, kinetic='jacobi',
             configs='singlet(1,1)',
             use_jastrow=True)

# sampler
sampler = Metropolis(nwalkers=500, nstep=1000, step_size=0.1,
                     nelec=wf.nelec, ndim=wf.ndim,
                     init=mol.domain('normal'),
                     move={'type': 'all-elec', 'proba': 'normal'})

# optimizer
opt = Adam(wf.parameters(), lr=0.005)

# scheduler
scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.75)

# solver
solver = SolverOrbital(wf=wf, sampler=sampler,
                       optimizer=opt, scheduler=scheduler)

pos, e, v = solver.single_point(ntherm=-1, ndecor=100)

# pos = solver.sample(ntherm=500, ndecor=10)
# obs = solver.sampling_traj(pos)
# plot_observable(obs, e0=-8., ax=None)

# optimize the wave function
# solver.configure(task='wf_opt', freeze=['mo', 'bas_exp'])
# solver.observable(['local_energy'])
# solver.run(10, loss='energy')
# plot_observable(solver.obs_dict, e0=-8.06)

# # optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')
# solver.save_traj('h2o_traj.xyz')

# # plot the data
# plot_observable(solver.obs_dict)
