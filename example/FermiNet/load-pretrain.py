# in this file we will look at the result of pretraining of the FermiNet network and compare to orbital configurations.
import torch 
from torch import optim

from qmctorch.wavefunction import Molecule, Orbital
from qmctorch.sampler import Metropolis
from qmctorch.utils import set_torch_double_precision
from qmctorch.utils import (plot_energy, plot_data)
from qmctorch.utils.plot_mo import Display_orbital
from qmctorch.wavefunction import WaveFunction
from qmctorch.wavefunction.wf_FermiNet import FermiNet
from qmctorch.solver.solver_FermiNet import SolverFermiNet

import numpy as np 
import time
import matplotlib.pyplot as plt


# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree
set_torch_double_precision()

ndim =3
nbatch = 4096
halfnbatch = int(nbatch/2)

# define the molecule
water = "example/single_point/water.xyz"
H2= 'H 0 0 -0.69; H 0 0 0.69'
mol = Molecule(atom=H2,
            calculator='pyscf',
            basis='sto-3g',
            unit='bohr')

# create the network
wf = FermiNet(mol, hidden_nodes_e=254, hidden_nodes_ee=32, L_layers=4, Kdet=1)

# choose a sampler for training
sampler_training = Metropolis(nwalkers=halfnbatch,
                    nstep=10, step_size=0.2,
                    ntherm=-1, ndecor=100,
                    nelec=wf.nelec, init=mol.domain('atomic'),
                    move={'type': 'all-elec', 'proba': 'normal'})

# choose a optimizer
opt = optim.Adam(wf.parameters(), lr=1E-3)

# initialize solver
solverFermi = SolverFermiNet(wf, optimizer = opt, sampler=sampler_training)

solverFermi.load_checkpoint("FermiNet_model.pth")

wf = solverFermi.wf
hf = Orbital(mol, configs = "ground_state", use_jastrow=False) 
dims = ["x","y","z"]
spins = ["up","down"] 
for spin in spins:
    for dim in dims:
        Display_orbital(hf._get_slater_matrices, hf, plane= dim, 
                                    plane_coord=0.0,
                                    title="Hartree Fock orbital",
                                    path="/home/breebaart/dev/QMCTorch/Figures_training/pyscf_H2O_"+spin+"_"+dim,
                                    orbital_ind=[0,0],
                                    spin = spin)
        Display_orbital(wf.compute_mo, wf, plane=dim,
                                    plane_coord=0.0,
                                    title = "Pretraining Fermi Net spin {} for H2O epoch: 2000".format(spin), 
                                    path="/home/breebaart/dev/QMCTorch/Figures_training/pretraining_H2O_2000_"+spin+"_"+dim,
                                    orbital_ind=[0,0],
                                    spin = spin)
