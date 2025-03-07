import torch
import numpy as np
from torch import optim

from qmctorch.scf import Molecule

from qmctorch.solver import Solver
from qmctorch.sampler import Metropolis, Langevin
from qmctorch.utils import set_torch_double_precision
from qmctorch.utils.plot_data import (plot_energy, plot_data, plot_walkers_traj)
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel

# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree

set_torch_double_precision()
torch.random.manual_seed(0)
np.random.seed(0)

# define the molecule
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
               calculator='pyscf',
               basis='sto-3g',
               unit='bohr')

# jastrow
jastrow = JastrowFactor(mol, PadeJastrowKernel)

# define the wave function
wf = SlaterJastrow(mol, kinetic='jacobi',
                   configs='single_double(2,2)',
                   jastrow=jastrow)

# sampler
# sampler = Hamiltonian(nwalkers=100, nstep=100, nelec=wf.nelec,
#                       step_size=0.1, L=30,
#                       ntherm=-1, ndecor=10,
#                       init=mol.domain('atomic'))

sampler = Langevin(nwalkers=1000, nstep=1000, nelec=wf.nelec, 
                     ntherm=0, ndecor=1,
                     step_size=0.05, 
                     init=mol.domain('atomic'))

sampler2 = Metropolis(nwalkers=1000, nstep=1000, nelec=wf.nelec, 
                     ntherm=0, ndecor=1,
                     step_size=0.05, 
                     init=mol.domain('atomic'))

# optimizer
lr_dict = [{'params': wf.jastrow.parameters(), 'lr': 1E-2},
           {'params': wf.ao.parameters(), 'lr': 1E-6},
           {'params': wf.mo.parameters(), 'lr': 2E-3},
           {'params': wf.fc.parameters(), 'lr': 2E-3}]
opt = optim.Adam(lr_dict, lr=1E-3)

# scheduler
scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.90)

# QMC solver
solver = Solver(wf=wf, sampler=sampler, optimizer=opt, scheduler=None)

# perform a single point calculation
# obs = solver.single_point()
obs = solver.sampling_traj()
# plot_walkers_traj(obs.local_energy, walkers='mean')

solver.sampler = sampler2
obs2 = solver.sampling_traj()

# configure the solver
# solver.configure(track=['local_energy', 'parameters'], freeze=['ao'],
#                  loss='energy', grad='manual',
#                  ortho_mo=False, clip_loss=False, clip_threshold=2,
#                  resampling={'mode': 'update',
#                              'resample_every': 1,
#                              'nstep_update': 150,
#                              'ntherm_update': 50}
#                  )

# pos = torch.rand(10, 6)
# pos.requires_grad = True

# wf.fc.weight.data = torch.rand(1, 4) - 0.5
# print(wf(pos))

# solver.evaluate_grad_manual(pos)
# print(wf.jastrow.jastrow_kernel.weight.grad)
# wf.zero_grad()


# solver.evaluate_grad_manual_3(pos)
# print(wf.jastrow.jastrow_kernel.weight.grad)
# wf.zero_grad()

# solver.evaluate_grad_auto(pos)
# print(wf.jastrow.jastrow_kernel.weight.grad)
# wf.zero_grad()


# optimize the wave function
# obs = solver.run(5)  # , batchsize=10)

# plot
# plot_energy(obs.local_energy, e0=-1.1645, show_variance=True)
# plot_data(solver.observable, obsname='jastrow.weight')
