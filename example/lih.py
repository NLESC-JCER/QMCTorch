from torch import optim
from torch.optim import Adam, SGD

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital
from deepqmc.solver.torch_utils import set_torch_double_precision

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.sampler.hamiltonian import Hamiltonian
from deepqmc.wavefunction.molecule import Molecule

from deepqmc.solver.plot_data import plot_observable
import matplotlib.pyplot as plt

set_torch_double_precision()

# define the molecule
mol = Molecule(atom='Li 0 0 0; H 0 0 3.015',
               basis_type='gto',
               basis='sto-6g',
               unit='bohr')

# define the wave function
wf = Orbital(mol, kinetic='jacobi',
             configs='ground_state',
             use_jastrow=True)

# sampler
sampler = Metropolis(nwalkers=5000, nstep=1000, step_size=0.05,
                     nelec=wf.nelec, ndim=wf.ndim,
                     init=mol.domain('normal'),
                     move={'type': 'all-elec-iter', 'proba': 'normal'})

# sampler = Hamiltonian(nwalkers=500, nstep=500,
#                       step_size=0.05, L=10,
#                       nelec=wf.nelec, ndim=wf.ndim,
#                       init=mol.domain('normal'))

# optimizer
lr_dict = [{'params': wf.jastrow.parameters(), 'lr': 1E-2},
           {'params': wf.ao.parameters(), 'lr': 1E-3},
           {'params': wf.mo.parameters(), 'lr': 1E-3},
           {'params': wf.fc.parameters(), 'lr': 1E-3}]


#opt = Adam(lr_dict, lr=1E-3)
opt = SGD(lr_dict, lr=1E-1, momentum=0.9)

# scheduler
scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.75)

# solver
solver = SolverOrbital(wf=wf, sampler=sampler,
                       optimizer=opt, scheduler=None)


# pos, e, v = solver.single_point(ntherm=-1, ndecor=100)
# eloc = solver.wf.local_energy(pos)
# plt.hist(eloc.detach().numpy(), bins=50)
# plt.show()

pos = solver.sample(ntherm=0, ndecor=10)
obs = solver.sampling_traj(pos)
plot_observable(obs, e0=-8., ax=None)

# optimize the wave function
# solver.configure(task='wf_opt', freeze=['ao'])
# solver.observable(['local_energy'])
# solver.initial_sampling(ntherm=-1, ndecor=100)
# solver.resampling(nstep=10, resample_every=1)
# solver.sampler.step_size = 1E-4
# solver.ortho_mo = True
# data = solver.run(50, loss='variance', clip_loss=True)
# plot_observable(solver.obs_dict, e0=-8.06)

# # optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')
# solver.save_traj('h2o_traj.xyz')

# # plot the data
# plot_observable(solver.obs_dict)
