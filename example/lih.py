from torch import optim
from torch.optim import Adam, SGD

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital
from deepqmc.utils.torch_utils import set_torch_double_precision

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.sampler.hamiltonian import Hamiltonian
from deepqmc.wavefunction.molecule import Molecule

from deepqmc.utils.plot_data import (load_observable,
                                     save_observalbe, plot_block,
                                     plot_walkers_traj,
                                     plot_energy, plot_data)

set_torch_double_precision()

# define the molecule
mol = Molecule(atom='Li 0 0 0; H 0 0 3.015', 
             calculator='pyscf',
              basis='dzp', 
              unit='bohr')

# define the wave function
wf = Orbital(mol, kinetic='jacobi',
             configs='ground_state',
             use_jastrow=True)
             
wf.jastrow.weight.data[0] = 1. 



# sampler
sampler = Metropolis(nwalkers=500, nstep=2000, step_size=0.05,
                     nelec=wf.nelec, ndim=wf.ndim,
                     init=mol.domain('atomic'),
                     move={'type': 'all-elec', 'proba': 'normal'},wf=wf)


# optimizer
lr_dict = [{'params': wf.jastrow.parameters(), 'lr': 1E-3},
           {'params': wf.ao.parameters(), 'lr': 1E-6},
           {'params': wf.mo.parameters(), 'lr': 1E-3},
           {'params': wf.fc.parameters(), 'lr': 1E-3}]


opt = Adam(lr_dict, lr=1E-3)
# opt = SGD(lr_dict, lr=1E-1, momentum=0.9)
# opt = StochasticReconfiguration(wf.parameters(), wf)

# scheduler
scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.85)

# solver
solver = SolverOrbital(wf=wf, sampler=sampler,
                       optimizer=opt, scheduler=None)

if 1:
    pos, e, v = solver.single_point(ntherm=-1, ndecor=100)
    # eloc = solver.wf.local_energy(pos)
    # plt.hist(eloc.detach().numpy(), bins=50)
    # plt.show()

    # pos = solver.sample(ntherm=0, ndecor=10)
    # obs = solver.sampling_traj(pos)
    # plot_energy(obs, e0=-8.)

if 0:

    solver.configure(task='wf_opt', freeze=['ao', 'mo'])
    solver.observable(['local_energy'])

    solver.initial_sampling(ntherm=1000, ndecor=100)
    solver.resampling(nstep=25, step_size=0.2,
                    resample_from_last=True,
                    resample_every=1, tqdm=False)

    solver.ortho_mo = False

    data = solver.run(50, batchsize=None,
                    loss='energy',
                    grad='manual',
                    clip_loss=True)

    save_observalbe('lih.pkl', solver.obs_dict)
    e, v = plot_energy(solver.obs_dict, e0=-8.06, show_variance=True)
    plot_data(solver.obs_dict, obs='jastrow.weight')
    

# # # optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')
# solver.save_traj('h2o_traj.xyz')

# # plot the data
# plot_energy(solver.obs_dict)
