import torch
from torch import optim
from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel
from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver
from qmctorch.utils import set_torch_double_precision
from qmctorch.wavefunction.jastrows.graph.mgcn_jastrow import MGCNJastrowFactor
set_torch_double_precision()

# define the molecule
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
               calculator='pyscf', basis='dzp', unit='bohr')

# jastrow
jastrow = JastrowFactor(mol, PadeJastrowKernel)


# jastrow
_jastrow = MGCNJastrowFactor(
            mol,
            ee_model_kwargs={"n_layers": 2, "feats": 4, "predictor_hidden_feats": 2, "cutoff": 5.0, "gap": 1.0},
            en_model_kwargs={"n_layers": 2, "feats": 4, "predictor_hidden_feats": 2, "cutoff": 5.0, "gap": 1.0},
        )


# define the wave function
wf = SlaterJastrow(mol, kinetic='jacobi',
                   configs='ground_state', 
                   jastrow=jastrow) #.gto2sto()

# sampler
sampler = Metropolis(nwalkers=100, nstep=10, step_size=0.25,
                     nelec=wf.nelec, ndim=wf.ndim, init=mol.domain('atomic'))

# optimizer
lr_dict = [{'params': wf.jastrow.parameters(), 'lr': 1E-3},
           {'params': wf.ao.parameters(), 'lr': 1E-6},
           {'params': wf.mo.parameters(), 'lr': 2E-3},
           {'params': wf.fc.parameters(), 'lr': 2E-3}]
opt = optim.Adam(lr_dict, lr=1E-3)


# solver
solver = Solver(wf=wf, sampler=sampler, optimizer=opt, scheduler=None)
solver.configure(track=['local_energy', 'parameters'], freeze=['ao'],
                 loss='energy', grad='manual',
                 ortho_mo=False, clip_loss=False,
                 resampling={'mode': 'update',
                             'resample_every': 1,
                             'ntherm_update': 5}
                 )

pos = torch.rand(10, 6)
pos.requires_grad = True

solver.wf.local_energy(pos)

# single point
# obs = solver.single_point()
obs = solver.run(5)
