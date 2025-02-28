import unittest

from qmctorch.ase import QMCTorch
from qmctorch.ase.optimizer import TorchOptimizer
from ase import Atoms
from ase.optimize import FIRE
import torch
import numpy as np


class TestASEcalculator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

        d = 0.70
        self.h2 = Atoms("H2", positions=[(0, 0, -d / 2), (0, 0, d / 2)])

        # instantiate the calc
        self.h2.calc = QMCTorch()

        # SCF options
        self.h2.calc.scf_options.calculator = "pyscf"
        self.h2.calc.scf_options.basis = "sto-3g"

        # WF options
        self.h2.calc.wf_options.configs = "single_double(2,2)"
        self.h2.calc.wf_options.orthogonalize_mo = False
        self.h2.calc.wf_options.gto2sto = True
        self.h2.calc.wf_options.jastrow.kernel_kwargs = {"w": 1.0}

        # sampler options
        self.h2.calc.sampler_options.nwalkers = 10
        self.h2.calc.sampler_options.nstep = 500
        self.h2.calc.sampler_options.step_size = 0.5
        self.h2.calc.sampler_options.ntherm = 400
        self.h2.calc.sampler_options.ndecor = 10

        # solver options
        self.h2.calc.solver_options.freeze = []
        self.h2.calc.solver_options.niter = 5
        self.h2.calc.solver_options.tqdm = False
        self.h2.calc.solver_options.grad = "manual"

        # options for the resampling
        self.h2.calc.solver_options.resampling.mode = "update"
        self.h2.calc.solver_options.resampling.resample_every = 1
        self.h2.calc.solver_options.resampling.ntherm_update = 10

        # Optimize the wave function
        self.h2.calc.initialize()

    def test_calculate_energy(self):
        self.h2.calc.calculate(properties=["energy"])

    def test_calculate_forces(self):
        self.h2.calc.calculate(properties=["forces"])

    def test_torch_optim(self):
        dyn = TorchOptimizer(
            self.h2,
            trajectory="traj.xyz",
            nepoch_wf_init=10,
            nepoch_wf_update=5,
            tqdm=False,
        )
        dyn.run(fmax=0.005, steps=2)

    def test_fire_optim(self):
        dyn = FIRE(self.h2, trajectory="traj.xyz")
        dyn.run(fmax=0.005, steps=2)
