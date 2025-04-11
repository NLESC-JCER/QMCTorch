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
h2 = Atoms('H2', positions=[(0, 0, -d/2), (0, 0, d/2)])

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
h2.calc.sampler_options.symmetry = Dinfh(axis='z')

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

