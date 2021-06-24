import torch
from torch import optim

from qmctorch.sampler import Metropolis
from qmctorch.scf import Molecule
from qmctorch.solver import SolverSlaterJastrow
from qmctorch.utils import (plot_block, plot_blocking_energy,
                            plot_correlation_coefficient, plot_walkers_traj)
from qmctorch.wavefunction import SlaterJastrow

torch.manual_seed(0)

# molecule
mol = Molecule(
    atom='H 0 0 -0.69; H 0 0 0.69',
    unit='bohr',
    calculator='pyscf',
    basis='sto-3g')

# wave function
wf = SlaterJastrow(mol, kinetic='auto',
                   configs='single(2,2)')

# sampler
sampler = Metropolis(
    nwalkers=1000,
    nstep=1000,
    ntherm=0,
    ndecor=1,
    step_size=0.5,
    ndim=wf.ndim,
    nelec=wf.nelec,
    init=mol.domain('normal'),
    move={
        'type': 'all-elec',
        'proba': 'normal'})

opt = optim.Adam(wf.parameters(), lr=0.01)

solver = SolverSlaterJastrow(wf=wf, sampler=sampler, optimizer=opt)

pos = solver.sampler(wf.pdf)
obs = solver.sampling_traj(pos)

plot_correlation_coefficient(obs.local_energy, method='both')
plot_walkers_traj(obs.local_energy)
plot_block(obs.local_energy)
plot_blocking_energy(obs.local_energy, block_size=10)
