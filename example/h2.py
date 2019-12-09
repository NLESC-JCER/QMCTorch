from torch.optim import Adam

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.sampler.hamiltonian import Hamiltonian
from deepqmc.wavefunction.molecule import Molecule
from deepqmc.solver.plot_data import plot_observable
# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree

# define the molecule
# mol = Molecule(atom='H 0 0 -0.37; H 0 0 0.37', basis_type='gto',
#                basis='sto-3g',unit='bohr')
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
               basis_type='gto', basis='sto-3g', unit='bohr')

# define the wave function
wf = Orbital(mol, kinetic_jacobi=False, use_projector=False)

# sampler
sampler = Metropolis(nwalkers=1000, nstep=300, step_size=0.25,
                     ndim=wf.ndim, nelec=wf.nelec, move='one')

sampler2 = Hamiltonian(nwalkers=1000, nstep=100, L=50, step_size=0.1,
                       ndim=wf.ndim, nelec=wf.nelec, move='one')

# optimizer
opt = Adam(wf.parameters(), lr=0.01)

# solver
solver = SolverOrbital(wf=wf, sampler=sampler, optimizer=opt)

pos = solver.sample(ntherm=0, ndecor=1)
obs = solver.sampling_traj(pos)
plot_observable(obs, e0=-1.16, ax=None)


# # optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')

# # plot the data
# plot_observable(solver.obs_dict,e0=-1.16)
