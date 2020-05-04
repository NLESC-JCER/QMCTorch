from torch import optim

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.solver import SolverOrbital
from qmctorch.sampler import Metropolis
from qmctorch.utils import set_torch_double_precision
from qmctorch.utils import (plot_energy, plot_data)

# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree

set_torch_double_precision()

# define the molecule
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
               calculator='adf',
               basis='dzp',
               unit='bohr')

# define the wave function
wf = Orbital(mol, kinetic='jacobi',
             configs='cas(2,2)',
             use_jastrow=True,
             cuda=True)

wf.jastrow.weight.data[0] = 1.

# sampler
sampler = Metropolis(nwalkers=2000,
                     nstep=2000, step_size=0.2,
                     ntherm=-1, ndecor=100,
                     nelec=wf.nelec, init=mol.domain('atomic'),
                     move={'type': 'all-elec', 'proba': 'normal'},
                    cuda=True)


# optimizer
lr_dict = [{'params': wf.jastrow.parameters(), 'lr': 3E-3},
           {'params': wf.ao.parameters(), 'lr': 1E-6},
           {'params': wf.mo.parameters(), 'lr': 1E-3},
           {'params': wf.fc.parameters(), 'lr': 2E-3}]
opt = optim.Adam(lr_dict, lr=1E-3)

# scheduler
scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.90)

# QMC solver
solver = SolverOrbital(wf=wf, sampler=sampler,
                       optimizer=opt, scheduler=None)

# perform a single point calculation
obs = solver.single_point()

# optimize the wave function
solver.configure(task='wf_opt', freeze=['ao', 'mo'])
solver.track_observable(['local_energy'])

solver.configure_resampling(mode='update',
                            resample_every=1,
                            nstep_update=50)
solver.ortho_mo = False
obs = solver.run(250, batchsize=None,
                 loss='energy',
                 grad='manual',
                 clip_loss=False)

plot_energy(obs.local_energy, e0=-1.1645, show_variance=True)
plot_data(solver.observable, obsname='jastrow.weight')
