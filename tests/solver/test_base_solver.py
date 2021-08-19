import unittest

import numpy as np
import torch
import torch.optim as optim


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

            # sample and compute observables
            obs = self.solver.single_point()
            e, v = obs.energy, obs.variance

            if self.expected_energy is not None:
                assert(
                    np.any(np.isclose(e.data.item(), np.array(self.expected_energy))))

            if self.expected_variance is not None:
                assert(
                    np.any(np.isclose(v.data.item(), np.array(self.expected_variance))))

        def test2_wf_opt_grad_auto(self):

            self.solver.configure(track=['local_energy', 'parameters'],
                                  loss='energy', grad='auto')
            _ = self.solver.run(5)

        def test3_wf_opt_grad_manual(self):

            self.solver.configure(track=['local_energy', 'parameters'],
                                  loss='energy', grad='manual')
            _ = self.solver.run(5)
