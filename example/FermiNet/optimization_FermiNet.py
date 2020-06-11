from torch import optim

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.solver import SolverOrbital
from qmctorch.sampler import Metropolis
from qmctorch.utils import set_torch_double_precision
from qmctorch.utils import (plot_energy, plot_data)
from qmctorch.utils.plot_mo import Display_orbital
from qmctorch.wavefunction.wf_FermiNet import FermiNet
from qmctorch.solver.solver_FermiNet import SolverFermiNet


# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree
pathloc ="/home/breebaart/dev/QMCTorch/TrainingExaminig/Training_H2_full/"

set_torch_double_precision()

# training hyperparameters
ndim =3
nbatch = 4096
halfnbatch = int(nbatch/2)

# define the molecule
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
               calculator='pyscf',
               basis='sto-3g',
               unit='bohr')

# create the network
wf = FermiNet(mol, hidden_nodes_e=254, hidden_nodes_ee=32, L_layers=4, Kdet=1)

# sampler for energy optimization
sampler_energy_opt = Metropolis(nwalkers=nbatch,
                     nstep=10, step_size=0.2,
                     ntherm=-1, ndecor=100,
                     nelec=wf.nelec, init=mol.domain('atomic'),
                     move={'type': 'all-elec', 'proba': 'normal'})

# sampler for pretraining
sampler_pretraining = Metropolis(nwalkers=halfnbatch,
                    nstep=10, step_size=0.2,
                    ntherm=-1, ndecor=100,
                    nelec=wf.nelec, init=mol.domain('atomic'),
                    move={'type': 'all-elec', 'proba': 'normal'})

# sampler for observations
sampler_observation  = Metropolis(nwalkers=200,
                     nstep=200, step_size=0.2,
                     ntherm=-1, ndecor=100,
                     nelec=wf.nelec, init=mol.domain('atomic'),
                     move={'type': 'all-elec', 'proba': 'normal'})

# optmizer
opt = optim.Adam(wf.parameters(), lr=1E-3)

# scheduler
scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.90)

# QMC solver
solver = SolverFermiNet(wf=wf, optimizer=opt, scheduler=scheduler)

# perform a single point calculation before pretraining
obs = solver.single_point(sampler=sampler_observation)

# # pretrain the FermiNet to hf orbitals
solver.pretrain(1000, optimizer=opt, sampler=sampler_pretrain)
solver.save_loss_list("pretrain_loss.pt")

solver.plot_loss(path=pathloc+ "pretraining_loss")

# load from pretrained Ferminet
# load = "example/FermiNet/FermiNet_model_pretrain_H2.pth"
# solver.load_checkpoint(load)
                    
# perform a single point calculation
obs = solver.single_point(sampler=sampler_observation)

# optimize the wave function
solver.track_observable(['local_energy'])

solver.configure_resampling(mode='update',
                            resample_every=1,
                            nstep_update=50)

obs = solver.run(10, batchsize=None,
                 loss='energy',
                 clip_loss=5,
                 grad="manual")

plot_energy(obs.local_energy, e0=-1.1645, show_variance=False, path=pathloc+"energy_"+mol.name)

wf = solver.wf
hf = Orbital(mol, configs = "ground_state", use_jastrow=False) 
dims = ["x","y","z"]
spins = ["up","down"] 
for spin in spins:
    for dim in dims:
        Display_orbital(hf._get_slater_matrices, hf, plane= dim, 
                                    plane_coord=0.0,
                                    title="Basis orbital",
                                    path=pathloc+mol.basis+"_"+ mol.name+"_"+spin+"_"+dim,
                                    orbital_ind=[0,0],
                                    spin = spin)
        Display_orbital(wf.compute_mo, wf, plane=dim,
                                    plane_coord=0.0,
                                    title = "Fermi Net spin {} orbital for {}".format(spin, mol.name), 
                                    path=pathloc+"FermiNet_"+mol.name+"_"+spin+"_"+dim,
                                    orbital_ind=[0,0],
                                    spin = spin)


obs = solver.single_point(sampler=sampler_observation)
