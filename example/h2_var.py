import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, SGD, lr_scheduler

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital
from deepqmc.solver.torch_utils import set_torch_double_precision
from deepqmc.sampler.metropolis import Metropolis
from deepqmc.optim.sr import StochasticReconfiguration

from deepqmc.wavefunction.molecule import Molecule
from deepqmc.solver.plot_data import plot_observable, load_observable, save_observalbe, plot_block_baby

# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree

set_torch_double_precision()

eloc = []
for i in range(20):

    # define the molecule
    mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
                   basis_type='sto',
                   basis='dz',
                   unit='bohr')

    # define the wave function
    wf = Orbital(mol, kinetic='jacobi',
                 configs='ground_state',
                 use_jastrow=True)

    wf.jastrow.weight.data[0] = 0.5

    # sampler
    sampler = Metropolis(nwalkers=1, nstep=1000, step_size=0.5,
                         ndim=wf.ndim, nelec=wf.nelec,
                         init=mol.domain('normal'),
                         move={'type': 'all-elec-iter', 'proba': 'normal'})

    # optimizer
    lr_dict = [{'params': wf.jastrow.parameters(), 'lr': 1E-3},
               {'params': wf.ao.parameters(), 'lr': 1E-6},
               {'params': wf.mo.parameters(), 'lr': 1E-3},
               {'params': wf.fc.parameters(), 'lr': 1E-3}]

    opt = Adam(lr_dict, lr=1E-3)
    # opt = SGD(lr_dict, lr=1E-1)
    # opt = StochasticReconfiguration(wf.parameters(), wf)

    # scheduler
    scheduler = lr_scheduler.StepLR(opt, step_size=100, gamma=0.90)

    # solver
    solver = SolverOrbital(wf=wf, sampler=sampler,
                           optimizer=opt, scheduler=None)

    pos, e, v = solver.single_point(ntherm=400, ndecor=1)
    eloc.append(e.detach().numpy())

eloc = np.array(eloc)
var = np.var(eloc)


# plot_block_baby(obs)
# eloc = obs['local_energy']
# eloc = np.array(eloc).squeeze()
# celoc = np.cumsum(eloc)/(np.arange(1100)+1)
# plt.plot(eloc, 'o')
# plt.plot(celoc)
# plt.show()

# optimize the wave function
# solver.configure(task='wf_opt', freeze=['ao'])

# solver.observable(['local_energy'])

# solver.initial_sampling(ntherm=-1, ndecor=100)

# solver.resampling(nstep=10, step_size=1E-4, resample_every=None, tqdm=False)

# solver.ortho_mo = True

# data = solver.run(500, batchsize=None,
#                   loss='weighted-energy',
#                   grad='manual',
#                   clip_loss=False)
# plot_observable(solver.obs_dict, e0=-1.1645)

# # optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')

# plot the data
# plot_observable(solver.obs_dict, e0=-1.16)
