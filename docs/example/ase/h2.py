from qmctorch.ase import QMCTorch     
from ase import Atoms 
from ase.optimize import GoodOldQuasiNewton
from ase.io import write
import torch
import numpy as np
from qmctorch.utils.plot_data import plot_walkers_traj, plot_correlation_coefficient

torch.random.manual_seed(0)
np.random.seed(0)

d = 0.74
h2 = Atoms('H2', positions=[(0, 0, -d/2), (0, 0, d/2)])

h2.calc = QMCTorch()

# SCF options
h2.calc.scf_options.calculator = 'pyscf'
h2.calc.scf_options.basis = 'sto-3g'

# WF options
h2.calc.wf_options.configs = 'ground_state'
# h2.calc.wf_options.configs = 'single_double(2,2)'
h2.calc.wf_options.orthogonalize_mo = False
# h2.calc.wf_options.gto2sto = True
h2.calc.wf_options.jastrow.kernel_kwargs = {'w':0.5}

# sampler options
h2.calc.sampler_options.nwalkers = 100
h2.calc.sampler_options.nstep  = 5000
h2.calc.sampler_options.step_size = 0.5
h2.calc.sampler_options.ntherm = 4000
h2.calc.sampler_options.ndecor = 10

# solver options
h2.calc.solver_options.freeze = []
h2.calc.solver_options.niter = 0
h2.calc.solver_options.tqdm = True
h2.calc.solver_options.grad = 'manual'

# options for the resampling
h2.calc.solver_options.resampling.mode = 'update'
h2.calc.solver_options.resampling.resample_every = 1
h2.calc.solver_options.resampling.ntherm_update = 100

# set solver
h2.calc.set_solver()

# sampling traj
obs = h2.calc.solver.sampling_traj()
plot_walkers_traj(obs.local_energy, walkers='mean')
plot_correlation_coefficient(obs.local_energy)

# sample
# obs = h2.calc.solver.single_point()



# h2.calc.solver.set_params_requires_grad(wf_params=False, geo_params=True)
# h2.calc.solver.evaluate_grad_auto(obs.pos)
# print(h2.calc.wf.ao.atom_coords.grad)

# h2.calc.wf.zero_grad()
# h2.calc.solver.evaluate_grad_manual(obs.pos)
# print(h2.calc.wf.ao.atom_coords.grad)


# h2.calc.wf.zero_grad()
# h2.calc.solver.evaluate_grad_manual_2(obs.pos)
# print(h2.calc.wf.ao.atom_coords.grad)



# # set param
# h2.calc.solver.set_params_requires_grad(wf_params=False, geo_params=True)

# compute energy
# h2.get_total_energy()

# compute the forces
# h2.get_forces()
# print(h2.get_forces())

# dyn = GoodOldQuasiNewton(h2, trajectory='traj.xyz')
# dyn.run(fmax=0.05, steps=2)
# write('final.xyz',h2)

# h2.calc.solver.evaluate_grad_auto(obs.)