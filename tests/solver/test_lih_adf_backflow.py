import unittest

import numpy as np
import torch
import torch.optim as optim

from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver
from qmctorch.scf import Molecule
from qmctorch.wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel
from qmctorch.wavefunction.orbitals.backflow import (
    BackFlowTransformation,
    BackFlowKernelInverse,
)
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.utils import set_torch_double_precision

from ..path_utils import PATH_TEST
from .test_base_solver import BaseTestSolvers


class TestLiHBackFlowADF(BaseTestSolvers.BaseTestSolverMolecule):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        set_torch_double_precision()

        # molecule
        path_hdf5 = (PATH_TEST / "hdf5/LiH_adf_dz.hdf5").absolute().as_posix()
        self.mol = Molecule(load=path_hdf5)

        # jastrow
        jastrow = JastrowFactor(self.mol, PadeJastrowKernel)

        # backflow
        backflow = BackFlowTransformation(self.mol, BackFlowKernelInverse)

        # wave function
        self.wf = SlaterJastrow(
            self.mol,
            kinetic="jacobi",
            jastrow=jastrow,
            backflow=backflow,
            configs="single_double(2,2)",
            include_all_mo=True,
        )

        # fc weights
        self.wf.fc.weight.data = torch.rand(self.wf.fc.weight.shape)

        # jastrow weights
        self.wf.jastrow.jastrow_kernel.weight.data = torch.rand(
            self.wf.jastrow.jastrow_kernel.weight.shape
        )

        # sampler
        self.sampler = Metropolis(
            nwalkers=500,
            nstep=200,
            step_size=0.05,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain("normal"),
            move={"type": "all-elec", "proba": "normal"},
        )

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        # solver
        self.solver = Solver(wf=self.wf, sampler=self.sampler, optimizer=self.opt)

        # artificial pos
        self.nbatch = 10
        self.pos = torch.as_tensor(np.random.rand(self.nbatch, self.wf.nelec * 3))
        self.pos.requires_grad = True


if __name__ == "__main__":
    unittest.main()
