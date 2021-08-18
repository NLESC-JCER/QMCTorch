import unittest
from torch.autograd import grad, gradcheck, Variable

import numpy as np
import torch


def hess(out, pos):
    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos,
                 grad_outputs=z,
                 only_inputs=True,
                 create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[1]):

        tmp = grad(jacob[:, idim], pos,
                   grad_outputs=z,
                   only_inputs=True,
                   create_graph=True)[0]

        hess[:, idim] = tmp[:, idim]

    return hess


class BaseTestJastrow:

    class ElecElecJastrowBaseTest(unittest.TestCase):

        def setUp(self) -> None:
            """Init the test case"""
            self.jastrow = None
            self.nbatch = None
            self.pos = None

        def test_jastrow(self):
            """simply checks that the values are not crashing."""
            val = self.jastrow(self.pos)

        def test_permutation(self):

            jval = self.jastrow(self.pos)

            # test spin up
            pos_xup = self.pos.clone()
            perm_up = list(range(self.nelec))
            perm_up[0] = 1
            perm_up[1] = 0
            pos_xup = pos_xup.reshape(self.nbatch, self.nelec, 3)
            pos_xup = pos_xup[:, perm_up, :].reshape(
                self.nbatch, self.nelec*3)

            jval_xup = self.jastrow(pos_xup)
            assert(torch.allclose(jval, jval_xup))

        def test_grad_distance(self):

            r = self.jastrow.edist(self.pos)
            dr = self.jastrow.edist(self.pos, derivative=1)
            dr_grad = grad(
                r,
                self.pos,
                grad_outputs=torch.ones_like(r))[0]
            gradcheck(self.jastrow.edist, self.pos)

            assert(torch.allclose(dr.sum(), dr_grad.sum(), atol=1E-5))

        def test_sum_grad_jastrow(self):

            val = self.jastrow(self.pos)
            dval = self.jastrow(self.pos, derivative=1)
            dval_grad = grad(
                val,
                self.pos,
                grad_outputs=torch.ones_like(val))[0]

            dval_grad = dval_grad.view(
                self.nbatch, self.nelec, 3).sum(2)
            gradcheck(self.jastrow, self.pos)

            assert torch.allclose(dval, dval_grad)
            assert(torch.allclose(dval.sum(), dval_grad.sum()))

        def test_grad_jastrow(self):

            val = self.jastrow(self.pos)
            dval = self.jastrow(
                self.pos, derivative=1, sum_grad=False)
            print(dval.shape)
            dval_grad = grad(
                val,
                self.pos,
                grad_outputs=torch.ones_like(val))[0]

            dval_grad = dval_grad.view(
                self.nbatch, self.nelec, 3)

            assert torch.allclose(dval, dval_grad.transpose(1, 2))
            assert(torch.allclose(dval.sum(), dval_grad.sum()))

        def test_hess_jastrow(self):

            val = self.jastrow(self.pos)
            d2val_grad = hess(val, self.pos)
            d2val = self.jastrow(self.pos, derivative=2)

            assert torch.allclose(d2val, d2val_grad.view(
                self.nbatch, self.nelec, 3).sum(2))

            assert(torch.allclose(d2val.sum(), d2val_grad.sum()))
