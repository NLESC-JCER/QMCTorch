import unittest
import numpy as np


class BaseTestSolvers:
    class BaseTestSolverMolecule(unittest.TestCase):
        def setUp(self):
            self.mol = None
            self.wf = None
            self.sampler = None
            self.opt = None
            self.solver = None
            self.pos = None
            self.expected_energy = None
            self.expected_variance = None

        def test1_single_point(self):
            """
            Test the single point calculation of the solver. The calculation is run two times.
            The first time, the calculation is run with all the walkers and the
            second time with half of the walkers.
            """
            self.solver.single_point()
            batchsize = int(self.solver.sampler.walkers.nwalkers / 2)
            self.solver.single_point(batchsize=batchsize)

        def test2_wf_opt_grad_auto(self):
            """
            Test the optimization of the wave function using autograd.
            The optimization is run for 5 epochs with all the walkers and then
            for 5 epochs with half the walkers.
            """
            self.solver.configure(
                track=["local_energy", "parameters"], loss="energy", grad="auto"
            )
            _ = self.solver.run(5)
            batchsize = int(self.solver.sampler.walkers.nwalkers / 2)
            _ = self.solver.run(5, batchsize=batchsize)

        def test3_wf_opt_grad_manual(self):
            """
            Test the optimization of the wave function using manual gradients.
            The optimization is run for 5 epochs with all the walkers and then
            for 5 epochs with half the walkers.
            """
            self.solver.configure(
                track=["local_energy", "parameters"], loss="energy", grad="manual"
            )
            _ = self.solver.run(5)
            batchsize = int(self.solver.sampler.walkers.nwalkers / 2)
            _ = self.solver.run(5, batchsize=batchsize)
