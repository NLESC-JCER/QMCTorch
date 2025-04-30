import unittest
import torch
from torch.autograd import Variable, grad, gradcheck
from qmctorch.utils.torch_utils import diagonal_hessian as hess


def hess_mixed_terms(out, pos):
    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos, grad_outputs=z, only_inputs=True, create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)
    nelec = pos.shape[1] // 3
    k = 0

    for ielec in range(nelec):
        ix = ielec * 3
        tmp = grad(
            jacob[:, ix], pos, grad_outputs=z, only_inputs=True, create_graph=True
        )[0]

        hess[:, k] = tmp[:, ix + 1]
        k = k + 1
        hess[:, k] = tmp[:, ix + 2]
        k = k + 1

        iy = ielec * 3 + 1
        tmp = grad(
            jacob[:, iy], pos, grad_outputs=z, only_inputs=True, create_graph=True
        )[0]

        hess[:, k] = tmp[:, iy + 1]
        k = k + 1

    return hess


class BaseTestAO:
    class BaseTestAOderivatives(unittest.TestCase):
        def setUp(self):
            def ao_callable(pos, derivative=0, sum_grad=False, sum_hess=False):
                """Callable for the AO"""
                return None

            self.ao = ao_callable
            self.pos = None

        def test_ao_deriv(self):
            ao = self.ao(self.pos)
            dao = self.ao(self.pos, derivative=1)
            dao_grad = grad(ao, self.pos, grad_outputs=torch.ones_like(ao))[0]

            gradcheck(self.ao, self.pos)
            assert torch.allclose(dao.sum(), dao_grad.sum())

        def test_ao_grad_sum(self):
            _ = self.ao(self.pos)
            dao_sum = self.ao(self.pos, derivative=1, sum_grad=True)
            dao = self.ao(self.pos, derivative=1, sum_grad=False)

            assert torch.allclose(dao_sum, dao.sum(-1))

        def test_ao_hess(self):
            ao = self.ao(self.pos)
            d2ao = self.ao(self.pos, derivative=2)
            d2ao_grad, _ = hess(ao, self.pos)
            assert torch.allclose(d2ao.sum(), d2ao_grad.sum())

        def test_ao_hess_sum(self):
            _ = self.ao(self.pos)
            d2ao_sum = self.ao(self.pos, derivative=2, sum_hess=True)
            d2ao = self.ao(self.pos, derivative=2, sum_hess=False)
            assert torch.allclose(d2ao_sum, d2ao.sum(-1))

        def test_ao_all(self):
            ao = self.ao(self.pos)
            dao = self.ao(self.pos, derivative=1, sum_grad=False)
            d2ao = self.ao(self.pos, derivative=2)
            ao_all, dao_all, d2ao_all = self.ao(self.pos, derivative=[0, 1, 2])

            assert torch.allclose(ao, ao_all)
            assert torch.allclose(dao, dao_all)
            assert torch.allclose(d2ao, d2ao_all)
