from qmctorch.ase import QMCTorch     
from qmctorch.ase.optimizer import TorchOptimizer
from qmctorch.sampler.symmetry import Cinfv, Dinfh
from ase import Atoms 
from ase.optimize import GoodOldQuasiNewton, FIRE
from ase.io import write
import torch
import numpy as np
from qmctorch.utils.plot_data import plot_walkers_traj, plot_correlation_coefficient

torch.random.manual_seed(0)
np.random.seed(0)


configs = (torch.tensor([[0],[1]]), torch.tensor([[0],[1]]))

d = 0.70
h2 = Atoms('LiH', positions=[(0, 0, 0), (0, 0, 3.14)])

h2.calc = QMCTorch()

# SCF options
h2.calc.scf_options.calculator = 'adf'
h2.calc.scf_options.basis = 'dzp'

# WF options
# h2.calc.wf_options.configs = 'ground_state'
h2.calc.wf_options.configs = 'single_double(2,4)'
# h2.calc.wf_options.configs = configs
h2.calc.wf_options.mix_mo = False
h2.calc.wf_options.orthogonalize_mo = False
# h2.calc.wf_options.gto2sto = True
h2.calc.wf_options.jastrow.kernel_kwargs = {'w':1.0}

# sampler options
h2.calc.sampler_options.nwalkers = 10000
h2.calc.sampler_options.nstep  = 500
h2.calc.sampler_options.step_size = 0.5
h2.calc.sampler_options.ntherm = -1
h2.calc.sampler_options.ndecor = 10
h2.calc.sampler_options.symmetry = None

# solver options
h2.calc.solver_options.freeze = []
h2.calc.solver_options.niter = 50
h2.calc.solver_options.tqdm = True
h2.calc.solver_options.grad = 'manual'

# options for the resampling
h2.calc.solver_options.resampling.mode = 'update'
h2.calc.solver_options.resampling.resample_every = 1
h2.calc.solver_options.resampling.ntherm_update = 100


# Optimize the wave function
h2.calc.initialize()

h2.get_potential_energy()

# single point
# obs = h2.calc.solver.single_point()
# pos = obs.pos 

# h2.calc.solver.evaluate_grad_manual(pos)
# # print(h2.calc.solver.wf.fc.weight.grad)
# print(h2.calc.solver.wf.ao.bas_exp.grad)
# h2.calc.solver.wf.zero_grad() 


# symm_pos = Dinfh(axis='z')(pos)
# h2.calc.solver.evaluate_grad_manual(symm_pos)
# # print(h2.calc.solver.wf.fc.weight.grad)
# print(h2.calc.solver.wf.ao.bas_exp.grad)
# h2.calc.solver.wf.zero_grad()

# wf = h2.calc.wf 
# pos = torch.rand(5,6)
# ao = wf.ao(pos)

# print(wf.mo_scf(ao))
# print(wf.mo(ao))

# print(h2.calc.wf.mo_scf.weight.data)
# print(h2.calc.wf.ao.bas_exp.data)
# mo_init = torch.clone(h2.calc.wf.mo_scf.weight.data)
# # compute forces
# h2.get_forces()
# h2.get_potential_energy()



# pos = torch.rand(2,6)
# sym_pos = h2.calc.sampler.symmetry(pos)
# h2.calc.wf.fc.weight.data = torch.rand(1, 16)
# print(h2.calc.wf.local_energy(pos))
# print(h2.calc.wf.local_energy(sym_pos))


# print(mo_init - h2.calc.wf.mo_scf.weight.data)

# use torch optim for the optimization
# dyn = TorchOptimizer(h2, 
#                      trajectory='traj.xyz', 
#                      nepoch_wf_init=50, 
#                      nepoch_wf_update=15, 
#                      tqdm=True)
# dyn = FIRE(h2, trajectory='traj.xyz')
# dyn.run(fmax=0.005, steps=5)
# write('final.xyz',h2)
