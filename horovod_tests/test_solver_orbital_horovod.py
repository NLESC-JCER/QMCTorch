import unittest

import horovod.torch as hvd
import numpy as np
import torch
from torch import optim

import sys

sys.path.append('/home/matthijs/esc/QMCTorch')
sys.path.append('/home/matthijs/esc/QMCTorch/qmctorch')

from qmctorch.sampler import Metropolis
from qmctorch.scf import Molecule
from qmctorch.solver import SolverOrbitalHorovod
from qmctorch.utils import set_torch_double_precision
from qmctorch.wavefunction import Orbital


class TestSolverOribitalHorovod(unittest.TestCase):
    def setUp(self):
        hvd.init()

        torch.manual_seed(101)
        np.random.seed(101)

        set_torch_double_precision()

        # molecule
        mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
                       calculator='pyscf', basis='sto-3g',
                       unit='bohr', rank=hvd.local_rank())

        self.wf = Orbital(mol, kinetic='jacobi',
                          configs='cas(2,2)',
                          use_jastrow=True, cuda=False)

        self.wf.jastrow.weight.data[0] = 1.

        self.sampler = Metropolis(nwalkers=20,
                                  nstep=20, step_size=0.2,
                                  ntherm=-1, ndecor=100,
                                  nelec=self.wf.nelec, init=mol.domain('atomic'),
                                  move={'type': 'all-elec', 'proba': 'normal'})

        lr_dict = [{'params': self.wf.jastrow.parameters(), 'lr': 3E-3},
                   {'params': self.wf.ao.parameters(), 'lr': 1E-6},
                   {'params': self.wf.mo.parameters(), 'lr': 1E-3},
                   {'params': self.wf.fc.parameters(), 'lr': 2E-3}]
        self.opt = optim.Adam(lr_dict, lr=1E-3)

    def test_single_point(self):
        solver = SolverOrbitalHorovod(wf=self.wf, sampler=self.sampler,
                                      optimizer=self.opt, rank=hvd.rank())

        obs = solver.single_point(with_tqdm=False)

        ref_energy = torch.tensor([-0.8895])
        ref_error = torch.tensor([0.0836])
        ref_local_energy = torch.tensor([[-1.5022],
                                         [-1.0112],
                                         [-0.8444],
                                         [-0.9291],
                                         [-1.4554],
                                         [-0.4428],
                                         [-0.2968],
                                         [-0.8322],
                                         [-0.6501],
                                         [-0.9306],
                                         [-1.5022],
                                         [-1.0112],
                                         [-0.8444],
                                         [-0.9291],
                                         [-1.4554],
                                         [-0.4428],
                                         [-0.2968],
                                         [-0.8322],
                                         [-0.6501],
                                         [-0.9306]])
        ref_variance = torch.tensor([0.1397])

        if hvd.rank() == 0:
            assert torch.isclose(obs.energy, ref_energy, 1E-1)
            assert np.isclose(obs.error, ref_error, 1E-1)
            assert torch.allclose(obs.local_energy, ref_local_energy, 1E-1)
            assert torch.isclose(obs.variance, ref_variance, 1E-1)


if __name__ == "__main__":
    t = TestSolverOribitalHorovod()
    t.setUp()
    t.test_single_point()
