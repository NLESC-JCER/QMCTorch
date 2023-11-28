from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel
from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver
from qmctorch.utils import plot_walkers_traj


# define the molecule
mol = Molecule(atom='water.xyz', unit='angs',
               calculator='pyscf', basis='sto-3g' , 
               name='water', redo_scf=True)

# jastrow
jastrow = JastrowFactor(mol, PadeJastrowKernel)

# define the wave function
wf = SlaterJastrow(mol, kinetic='jacobi',
                   configs='ground_state', jastrow=jastrow)

# sampler
sampler = Metropolis(nwalkers=1000, nstep=500, step_size=0.25,
                     nelec=wf.nelec, ndim=wf.ndim,
                     init=mol.domain('atomic'),
                     move={'type': 'all-elec', 'proba': 'normal'})

# solver
solver = Solver(wf=wf, sampler=sampler)

# single point
obs = solver.single_point()

# reconfigure sampler
solver.sampler.ntherm = 0
solver.sampler.ndecor = 5

# compute the sampling traj
pos = solver.sampler(solver.wf.pdf)
obs = solver.sampling_traj(pos)
plot_walkers_traj(obs.local_energy, walkers='mean')
