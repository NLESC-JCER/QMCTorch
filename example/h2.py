import torch
from torch.optim import Adam
from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital
from deepqmc.solver.torch_utils import set_torch_double_precision
from deepqmc.sampler.metropolis import Metropolis
from deepqmc.optim.sr import StochasticReconfiguration as SR

from deepqmc.wavefunction.molecule import Molecule
from deepqmc.solver.plot_data import plot_observable

# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree

# set_torch_double_precision()

# define the molecule
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
               basis_type='gto', basis='sto-6g', unit='bohr')

# define the wave function
wf = Orbital(mol, kinetic='jacobi',
             configs='singlet(1,1)',
             use_jastrow=True)

# sampler
sampler = Metropolis(nwalkers=1000, nstep=1000, step_size=0.5,
                     ndim=wf.ndim, nelec=wf.nelec,
                     init=mol.domain('normal'),
                     move={'type': 'all-elec', 'proba': 'normal'})

# optimizer
opt = Adam(wf.parameters(), lr=0.01)
#opt = SR(wf.parameters(), wf=wf, tau=0.01)

# solver
solver = SolverOrbital(wf=wf, sampler=sampler, optimizer=opt)
pos, _, _ = solver.single_point()

# pos = solver.sample(ntherm=0, ndecor=10)
# obs = solver.sampling_traj(pos)
# plot_observable(obs, e0=-1.16, ax=None)

# optimize the wave function
solver.configure(task='wf_opt', freeze=['mo', 'bas_exp'])
solver.observable(['local_energy'])
solver.run(5, loss='energy')

# # optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')

# plot the data
# plot_observable(solver.obs_dict, e0=-1.16)
