import sys
from torch.optim import Adam

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital 
from deepqmc.solver.plot_mol import plot_molecule, plot_molecule_mayavi

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.wavefunction.molecule import Molecule

# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree


# define the molecule
mol = Molecule(atom='H 0 0 -0.5; H 0 0 0.5', basis_type='sto', basis='sz')
#mol = Molecule(atom='H 0 0 -0.37; H 0 0 0.37', basis_type='gto', basis='sto-3g')

# define the wave function
wf = Orbital(mol)

#sampler
sampler = Metropolis(nwalkers=1000, nstep=100, step_size = 0.5, 
                     ndim = wf.ndim, nelec = wf.nelec, move = 'one')

# optimizer
opt = Adam(wf.parameters(),lr=0.01)

# solver
solver = SolverOrbital(wf=wf,sampler=sampler,optimizer=opt)
# pos,e,s = solver.single_point()
# print('Energy   : ', e)
# print('Variance : ', s)

plot_molecule_mayavi(solver)










