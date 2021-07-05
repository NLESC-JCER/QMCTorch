from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow
from qmctorch.sampler import Metropolis
from qmctorch.solver import SolverSlaterJastrow
from qmctorch.utils import plot_walkers_traj
from qmctorch.utils import set_torch_double_precision
set_torch_double_precision()

# define the molecule
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
               calculator='pyscf', basis='dzp', unit='bohr')


# define the wave function
wf = SlaterJastrow(mol, kinetic='jacobi',
                   configs='ground_state').gto2sto()

# sampler
sampler = Metropolis(nwalkers=1000, nstep=1000, step_size=0.25,
                     nelec=wf.nelec, ndim=wf.ndim,
                     init=mol.domain('atomic'),
                     move={'type': 'one-elec', 'proba': 'normal'},
                     logspace=False)


pos = sampler(wf.pdf)
e, s, err = wf._energy_variance_error(pos)

# print data
print('  Energy   : %f +/- %f' %
      (e.detach().item(), err.detach().item()))
print('  Variance : %f' % s.detach().item())

# # solver
# solver = SolverSlaterJastrow(wf=wf, sampler=sampler)

# # single point
# obs = solver.single_point(logspace=True)

# # reconfigure sampler
# solver.sampler.ntherm = 0
# solver.sampler.ndecor = 5

# # compute the sampling traj
# pos = solver.sampler(solver.wf.pdf)
# obs = solver.sampling_traj(pos)
# plot_walkers_traj(obs.local_energy, walkers='mean')
