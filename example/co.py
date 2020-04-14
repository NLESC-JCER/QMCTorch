from torch import optim

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.sampler import Metropolis
from qmctorch.solver import SolverOrbital
from qmctorch.utils import set_torch_double_precision

set_torch_double_precision()

# define the molecule
mol = Molecule(atom='C 0 0 0; O 0 0 2.173',
               calculator='pyscf',
               basis='sto-3g',
               unit='bohr')


# define the wave function
wf = Orbital(mol, kinetic='jacobi',
             configs='cas(4,4)',
             use_jastrow=True)

# sampler
sampler = Metropolis(nwalkers=1000, nstep=1000, step_size=0.1,
                     nelec=wf.nelec, ndim=wf.ndim,
                     init=mol.domain('normal'),
                     move={'type': 'one-elec', 'proba': 'normal'})

# optimizer
opt = optim.Adam(wf.parameters(), lr=0.005)

# scheduler
scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.75)

# solver
solver = SolverOrbital(wf=wf, sampler=sampler,
                       optimizer=opt, scheduler=scheduler)

# # single point
# pos, e, v = solver.single_point(ntherm=500, ndecor=100)

# #  sampling traj
# pos = solver.sample(ntherm=0, ndecor=10)
# obs = solver.sampling_traj(pos)
# plot_energy(obs, e0=-111.)


# optimize the wave function
# solver.configure(task='wf_opt', freeze=['bas_exp'])
# solver.observable(['local_energy'])
# solver.run(10, loss='energy')
# plot_energy(solver.obs_dict, e0=-113.)


# # optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')
# solver.save_traj('h2o_traj.xyz')

# # plot the data
# plot_energy(solver.obs_dict)
