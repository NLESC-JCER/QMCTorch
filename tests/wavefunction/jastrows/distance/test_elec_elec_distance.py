import torch
from torch.autograd import Variable, grad
from qmctorch.wavefunction.jastrows.distance import ElectronElectronDistance
import unittest
from qmctorch.utils import set_torch_double_precision

set_torch_double_precision()


class TestElecElecDistance(unittest.TestCase):
    def setUp(self):
        self.nup, self.ndown = 1, 1
        self.nelec = self.nup + self.ndown
        self.nbatch = 5

        self.pos = torch.rand(self.nbatch, self.nelec * 3)
        self.pos.requires_grad = True

        self.edist = ElectronElectronDistance(self.nelec)

    def test_grad_distance(self):
        """test computation the gradient of the distance.

        Note that the edist function does not compute all the derivative terms
        Instead when calling dr = edis(pos,1) -> Nbatch x 3 x Nelec x Nelec
        with : dr(:,0,i,j) = d/d x_i r_{ij}
               dr(:,1,i,j) = d/d y_i r_{ij}
               dr(:,2,i,j) = d/d z_i r_{ij}

        Autograd will compute  drr_grad[i] = d/d_xi (sum_{a,b}  r_{ab})

        Therefore the edist(pos,1) misses (on purposes) the terms d/dx_i r_{ki}
        However we always have

        d r_{ij} / d x_i  = d r_{ij} / d x_j (= - d r_{ji} / d x_i)
        """

        # elec-elec distance
        r = self.edist(self.pos)

        # derivative of r wrt the first elec d/dx_i r_{ij}
        di_r = self.edist(self.pos, derivative=1)

        # derivative of r wrt the second elec d/dx_j r_{ij} (see notes)
        dj_r = di_r

        # sum
        dr = di_r + dj_r

        # compute the der with autograd
        dr_grad = grad(r, self.pos, grad_outputs=torch.ones_like(r))[0]

        # check sum
        assert torch.allclose(dr.sum(), dr_grad.sum(), atol=1e-5)

        # see the notes for the explanation of the factor 2
        dr = dr.sum(-1).permute(0, 2, 1).reshape(5, -1)
        assert torch.allclose(dr, dr_grad)


if __name__ == "__main__":
    # unittest.main()
    t = TestElecElecDistance()
    t.setUp()
    t.test_grad_distance()
