import numpy as np

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.wavefunction.molecule import Molecule


def rot_mat(angles):
    dA = (angles[1]-angles[0])*np.pi/180
    c, s = np.cos(dA), np.sin(dA)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def bend_molecule(mol, r, index=1):
    mol.atom_coords[index] = (r @ np.array(
        mol.atom_coords[index])).astype('float32')

    return mol


# define the molecule
mol = Molecule(atom='water_line.xyz', unit='angs',
               basis_type='gto', basis='sto-3g')

# define the wave function
wf = Orbital(mol, kinetic_jacobi=True)


# sampler
sampler = Metropolis(nwalkers=1000, nstep=5000, step_size=0.5,
                     ndim=wf.ndim, nelec=wf.nelec, move='one')

# solver
solver = SolverOrbital(wf=wf, sampler=sampler)
pos, e, v = solver.single_point()
sampler.nstep = 500

angles = np.linspace(0, 90, 10)
R = rot_mat(angles)

for iA in range(len(angles)):

    # define the wave function
    wf = Orbital(mol, kinetic_jacobi=True)
    solver.wf = wf

    pos, e, v = solver.single_point(pos=pos)

    # bend the mol
    mol = bend_molecule(mol, R)
