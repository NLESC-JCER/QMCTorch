# this file will contain the function for pretraining_steps of the FermiNet.
# the FermiNet is pretrained to pyscf sto-3g hf orbitals.
# this pretraining reduces the variance of the calculations when optimizing the FermiNet
# and allows to skip the more non-physical regions in the optimization.

import torch 
from torch import optim

from qmctorch.wavefunction import Molecule
from qmctorch.sampler import Metropolis
from qmctorch.utils import set_torch_double_precision
from qmctorch.utils import (plot_energy, plot_data)
from qmctorch.wavefunction import WaveFunction
from qmctorch.wavefunction.wf_FermiNet import FermiNet
from qmctorch.solver.pretraining_FermiNet import SolverFermiNet

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
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
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
solverFermi = SolverFermiNet(wf,sampler=sampler_training)

# pretrain the FermiNet to hf orbitals
solverFermi.pretrain(1000,optimizer=opt)

solverFermi.save_loss_list("pretrain_loss.pt")

solverFermi.plot_loss(path="/home/breebaart/dev/QMCTorch/TrainingExaminig/Pretraining/pretraining_loss")