import torch
from torch.autograd import Variable, grad


class TestDerivative():
    def __init__(self):
        self.nbatch = 5
        self.nelec = 2
        self.nbas = 2
        self.ndim = 3

        self.bas_coords = torch.tensor([[1., 0., 0.], [-1., 0., 0.]])
        self.bas_n = torch.tensor([1, 1])
        self.bas_exp = torch.tensor([1., 2.])

    def getA(self, input, derivative=0):

        xyz = (input.view(-1, self.nelec, 1, self.ndim) -
               self.bas_coords[None, ...])
        r = torch.sqrt((xyz**2).sum(3))
        R = self.radial(r, xyz, derivative=derivative)
        return R

    def radial(self, R, xyz=None, derivative=0):

        model = 2

        if derivative == 0:
            if model == 0:
                return R**self.bas_n
            if model == 1:
                return torch.exp(-self.bas_exp*R**2)
            if model == 2:
                return R**self.bas_n * torch.exp(-self.bas_exp*R**2)

        elif derivative > 0:

            sum_xyz = xyz.sum(3)

            rn = R**(self.bas_n)
            nabla_rn = (self.bas_n * R**(self.bas_n-2)).unsqueeze(-1) * xyz

            er = torch.exp(-self.bas_exp*R**2)
            nabla_er = -2*(self.bas_exp * er).unsqueeze(-1) * xyz

            if derivative == 1:

                nabla_rn = nabla_rn.sum(3)
                nabla_er = nabla_er.sum(3)

                if model == 0:
                    return nabla_rn
                if model == 1:
                    return nabla_er
                if model == 2:
                    return nabla_rn*er + rn*nabla_er

            elif derivative == 2:

                if model == 0:
                    return lap_rn

                if model == 1:
                    return lap_er

                if model == 2:
                    return lap_rn*er + 2*(nabla_rn*nabla_er).sum(3) + rn*lap_er

    def second_der_numeric(self, xyz, eps=1E-2):

        ndim = xyz.shape[1]

        a0 = self.getA(xyz)
        out = torch.zeros(a0.shape)

        for idim in range(ndim):

            xyzp = xyz.clone()
            xyzp[:, idim] += eps
            ap = self.getA(xyzp)

            xyzm = xyz.clone()
            xyzm[:, idim] -= eps
            am = self.getA(xyzm)

            out += (am+ap-2*a0)/eps**2

        return out

    def first_der_autograd(self, x):

        out = self.getA(x)
        #out = torch.det(out)

        z = Variable(torch.ones(out.shape))
        jacob = grad(out, x,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]

        return jacob

    def second_der_autograd(self, xyz):

        out = self.getA(xyz)

        # compute the jacobian
        z = Variable(torch.ones(out.shape))
        jacob = grad(out, xyz,
                     grad_outputs=z,
                     allow_unused=True,
                     create_graph=True)[0]

        # compute the diagonal element of the Hessian
        z = Variable(torch.ones(jacob.shape[0]))
        hess = torch.zeros(jacob.shape)

        for idim in range(jacob.shape[1]):
            tmp = grad(jacob[:, idim], xyz,
                       grad_outputs=z,
                       retain_graph=True,
                       only_inputs=True,
                       allow_unused=True)[0]

            hess[:, idim] = tmp[:, idim]

        return hess


Test = TestDerivative()

xyz = Variable(torch.rand(Test.nbatch, 3*Test.nelec))
xyz.requires_grad = True


dAO = Test.getA(xyz, derivative=1)
dAO_auto = Test.first_der_autograd(xyz)
print(dAO.sum())
print(dAO_auto.sum())

d2AO = Test.getA(xyz, derivative=2)
d2AO_num = Test.second_der_numeric(xyz)
d2AO_auto = Test.second_der_autograd(xyz)

print(d2AO.sum())
print(d2AO_auto.sum())
print(d2AO_num.sum())
