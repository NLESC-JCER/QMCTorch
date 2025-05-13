import unittest

import numpy as np
import torch
import torch.optim as optim
from .test_base_solver import BaseTestSolvers
from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver
from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.wavefunction.jastrows.elec_elec import (
    JastrowFactor as JastrowFactorElecElec,
    SpinPairFullyConnectedJastrowKernel as ElecElecKernel,
)
from qmctorch.wavefunction.jastrows.elec_nuclei import (
    JastrowFactor as JastrowFactorElecNuclei,
    PadeJastrowKernel as ElecNucleiKernel,
)
from qmctorch.wavefunction.jastrows.elec_elec_nuclei import (
    BoysHandyJastrowKernel as ElecElecNucleiKernel,
    JastrowFactor as JastrowFactorElecElecNuc,
)

__PLOT__ = True


class TestH2SamplerHMC(BaseTestSolvers.BaseTestSolverMolecule):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

        # molecule
        self.mol = Molecule(
            atom="H 0 0 -0.69; H 0 0 0.69",
            unit="bohr",
            calculator="pyscf",
            basis="sto-3g",
        )

        # jastrow
        jastrow_ee = JastrowFactorElecElec(self.mol, ElecElecKernel)
        jastrow_en = JastrowFactorElecNuclei(self.mol, ElecNucleiKernel)
        jastrow_een = JastrowFactorElecElecNuc(self.mol, ElecElecNucleiKernel)

        # wave function
        self.wf = SlaterJastrow(
            self.mol,
            kinetic="jacobi",
            configs="single(2,2)",
            jastrow=[jastrow_ee, jastrow_en, jastrow_een],
        )

        self.sampler = Metropolis(
            nwalkers=1000,
            nstep=2000,
            step_size=0.5,
            ndim=self.wf.ndim,
            nelec=self.wf.nelec,
            init=self.mol.domain("normal"),
            move={"type": "all-elec", "proba": "normal"},
        )

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        # solver
        self.solver = Solver(wf=self.wf, sampler=self.sampler, optimizer=self.opt)


if __name__ == "__main__":
    unittest.main()
