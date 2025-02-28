import torch
from torch import optim
import horovod.torch as hvd

from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow
from qmctorch.solver import SolverMPI
from qmctorch.sampler import Metropolis
from qmctorch.utils import set_torch_double_precision
from qmctorch.utils import plot_energy, plot_data

# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree

hvd.init()
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(hvd.rank())

set_torch_double_precision()

# define the molecule
mol = Molecule(
    atom="H 0 0 -0.69; H 0 0 0.69",
    unit="bohr",
    calculator="pyscf",
    basis="sto-3g",
    rank=hvd.local_rank(),
    mpi_size=hvd.local_size(),
)


# define the wave function
wf = SlaterJastrow(mol, kinetic="jacobi", configs="cas(2,2)", cuda=use_cuda)

# sampler
sampler = Metropolis(
    nwalkers=200,
    nstep=200,
    step_size=0.2,
    ntherm=-1,
    ndecor=100,
    nelec=wf.nelec,
    init=mol.domain("atomic"),
    move={"type": "all-elec", "proba": "normal"},
    cuda=use_cuda,
)


# optimizer
lr_dict = [
    {"params": wf.jastrow.parameters(), "lr": 3e-3},
    {"params": wf.ao.parameters(), "lr": 1e-6},
    {"params": wf.mo.parameters(), "lr": 1e-3},
    {"params": wf.fc.parameters(), "lr": 2e-3},
]
opt = optim.Adam(lr_dict, lr=1e-3)

# scheduler
scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.90)

# QMC solver
solver = SolverMPI(
    wf=wf, sampler=sampler, optimizer=opt, scheduler=scheduler, rank=hvd.rank()
)

# configure the solver
solver.configure(
    track=["local_energy"],
    freeze=["ao", "mo"],
    loss="energy",
    grad="auto",
    ortho_mo=False,
    clip_loss=False,
    resampling={"mode": "update", "resample_every": 1, "nstep_update": 50},
)

# optimize the wave function
obs = solver.run(250)

if hvd.rank() == 0:
    plot_energy(obs.local_energy, e0=-1.1645, show_variance=True)
    plot_data(solver.observable, obsname="jastrow.weight")
