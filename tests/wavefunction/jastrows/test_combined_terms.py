import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch.autograd import Variable, grad, gradcheck

from qmctorch.wavefunction.jastrows.jastrow_factor_combined_terms import (
    JastrowFactorCombinedTerms,
)
from qmctorch.wavefunction.jastrows.elec_elec.kernels import (
    PadeJastrowKernel as PadeJastrowKernelElecElec,
)
from qmctorch.wavefunction.jastrows.elec_nuclei.kernels import (
    PadeJastrowKernel as PadeJastrowKernelElecNuc,
)
from qmctorch.wavefunction.jastrows.elec_elec_nuclei.kernels import (
    BoysHandyJastrowKernel,
)

torch.set_default_tensor_type(torch.DoubleTensor)


def hess(out, pos):
    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos, grad_outputs=z, only_inputs=True, create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[1]):
        tmp = grad(
            jacob[:, idim], pos, grad_outputs=z, only_inputs=True, create_graph=True
        )[0]

        hess[:, idim] = tmp[:, idim]

    return hess


class TestJastrowCombinedTerms(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

        self.nup, self.ndown = 4, 4
        self.nelec = self.nup + self.ndown
        self.atoms = np.random.rand(4, 3)

        self.mol = SimpleNamespace(
            nup=self.nup, ndown=self.ndown, atom_coords=self.atoms
        )

        self.jastrow = JastrowFactorCombinedTerms(
            self.mol,
            jastrow_kernel={
                "ee": PadeJastrowKernelElecElec,
                "en": PadeJastrowKernelElecNuc,
                "een": BoysHandyJastrowKernel,
            },
            jastrow_kernel_kwargs={"ee": {"w": 1.0}, "en": {"w": 1.0}, "een": {}},
        )

        self.nbatch = 5

        self.pos = torch.rand(self.nbatch, self.nelec * 3)
        self.pos.requires_grad = True

    def test_jastrow(self):
        val = self.jastrow(self.pos)

    def test_grad_jastrow(self):
        val = self.jastrow(self.pos)
        dval = self.jastrow(self.pos, derivative=1)
        dval_grad = grad(val, self.pos, grad_outputs=torch.ones_like(val))[0]

        dval_grad = dval_grad.view(self.nbatch, self.nelec, 3).sum(2)
        gradcheck(self.jastrow, self.pos)

        assert torch.allclose(dval, dval_grad)
        assert torch.allclose(dval.sum(), dval_grad.sum())

    def test_hess_jastrow(self):
        val = self.jastrow(self.pos)
        d2val_grad = hess(val, self.pos)
        d2val = self.jastrow(self.pos, derivative=2)

        assert torch.allclose(d2val.sum(), d2val_grad.sum())
        assert torch.allclose(d2val, d2val_grad.view(self.nbatch, self.nelec, 3).sum(2))


if __name__ == "__main__":
    unittest.main()
    # t = TestJastrowCombinedTerms()
    # t.setUp()
    # t.test_hess_jastrow()
