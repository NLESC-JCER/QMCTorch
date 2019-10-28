import sys
import torch
from torch.autograd import Variable
from torch.optim import Adam

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital
#from deepqmc.solver.solver_orbital_distributed import DistSolverOrbital  as SolverOrbital

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.wavefunction.molecule import Molecule

from deepqmc.solver.plot_orbital import plot_molecule
from deepqmc.solver.plot_orbital import plot_molecule_mayavi as plot_molecule
from deepqmc.solver.plot_data import plot_observable


# define the molecule
mol = Molecule(atom='Li 0 0 0; H 0 0 3.015', 
               basis_type='gto', 
               basis='sto-3g', 
               unit='bohr')


# define the wave function
wf = Orbital(mol)

#sampler
sampler = Metropolis(nwalkers=1000, nstep=500, step_size = 0.5, 
                     ndim = wf.ndim, nelec = wf.nelec, move = 'one')

# optimizer
opt = Adam(wf.parameters(),lr=0.01)

# solver
solver = SolverOrbital(wf=wf,sampler=sampler,optimizer=opt)
pos = Variable(torch.rand(100,mol.nelec*3))
pos.requires_grad = True
wf.ao(pos)

#pos = solver.single_point()

# plot the molecule
#plot_molecule(solver)

# optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')

# plot the data
# plot_observable(solver.obs_dict,e0=-1.16)











