import sys
import torch
from torch.autograd import Variable
from torch.optim import Adam

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.wavefunction.molecule import Molecule

from deepqmc.solver.plot_orbital import plot_molecule
from deepqmc.solver.plot_orbital import plot_molecule_mayavi as plot_molecule
from deepqmc.solver.plot_data import plot_observable

# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree

# define the molecule
#mol = Molecule(atom='H 0 0 -0.37; H 0 0 0.37', basis_type='gto', basis='sto-3g',unit='bohr')
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69', basis_type='gto', basis='sto-3g',unit='bohr')

# define the wave function
wf = Orbital(mol,kinetic_jacobi=False)

#sampler
sampler = Metropolis(nwalkers=1000, nstep=1000, step_size = 0.5, 
                     ndim = wf.ndim, nelec = wf.nelec, move = 'one')

# optimizer
opt = Adam(wf.parameters(),lr=0.01)

# solver
solver = SolverOrbital(wf=wf,sampler=sampler,optimizer=opt)
pos = solver.single_point()

# # plot the molecule
#plot_molecule(solver)

# # optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')

# # plot the data
# plot_observable(solver.obs_dict,e0=-1.16)











