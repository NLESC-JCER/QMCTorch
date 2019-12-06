from torch.optim import Adam

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital


from deepqmc.sampler.metropolis import Metropolis
from deepqmc.wavefunction.molecule import Molecule

from deepqmc.solver.plot_data import plot_observable


# define the molecule
mol = Molecule(atom='Li 0 0 0; H 0 0 3.015',
               basis_type='gto',
               basis='sto-6g',
               unit='bohr')


# define the wave function
wf = Orbital(mol, kinetic_jacobi=True)

# sampler
sampler = Metropolis(nwalkers=1000, nstep=1000, step_size=0.5,
                     ndim=wf.ndim, nelec=wf.nelec, move='one')

# optimizer
opt = Adam(wf.parameters(), lr=0.01)

# solver
solver = SolverOrbital(wf=wf, sampler=sampler, optimizer=opt)


pos = solver.single_point()

# optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')

# plot the data
# plot_observable(solver.obs_dict,e0=-1.16)
