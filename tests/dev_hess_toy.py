import torch
from torch.autograd import grad, gradcheck, Variable


def hess(out, pos):
    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos,
                 grad_outputs=z,
                 only_inputs=True,
                 create_graph=True)[0]

    # compute the diagonal element of the Hessian
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[0]):

        tmp = grad(jacob[idim], pos,
                   only_inputs=True,
                   create_graph=True)[0]

        hess[idim] = tmp[idim]

    return hess


class PHI(object):

    def __init__(self):
        pass

    @staticmethod
    def phi0(x, der):

        if der == 0:
            return 8 * x**2 + 1
        if der == 1:
            return 16 * x
        if der == 2:
            return 16.

    @staticmethod
    def phi1(x, der):

        if der == 0:
            return 7 * x**2 + 4
        if der == 1:
            return 14 * x
        if der == 2:
            return 14.

    def __call__(self, x, der=0):
        if der == 0:

            return (x**2).view(1, 2).repeat(2, 1).transpose(
                0, 1) * torch.tensor([8, 7]) + torch.tensor([1., 4.])
        else:
            return torch.tensor([
                [self.phi0(x[0], der), self.phi1(x[0], der)],
                [self.phi0(x[1], der), self.phi1(x[1], der)]
            ])


class JAST(object):

    def __init__(self):
        pass

    @staticmethod
    def jast0(x, der):
        x0, x1 = x

        if der == 0:
            return x0**2 * x1**2
        if der == 1:
            return 2*x0 * x1**2, x0**2 * 2*x1
        if der == 2:
            return 2 * x1**2, x0**2 * 2

    @staticmethod
    def jast1(x, der):

        x0, x1 = x

        if der == 0:
            return 4 * x0**2 * x1**2
        if der == 1:
            return 4 * 2 * x0 * x1**2, 4 * x0**2 * 2 * x1
        if der == 2:
            return 8 * x1**2, 8 * x0**2

    def __call__(self, x, der=0):

        if der == 0:
            return (x**2).prod().view(1).repeat(2) * torch.tensor([1, 4])

        if der == 1:
            djast0 = self.jast0(x, der)
            djast1 = self.jast1(x, der)
            return torch.tensor([
                [djast0[0], djast1[0]],
                [djast0[1], djast1[1]]
            ])

        if der == 2:
            djast0 = self.jast0(x, der)
            djast1 = self.jast1(x, der)
            return torch.tensor([
                [djast0[0], djast1[0]],
                [djast0[1], djast1[1]]
            ])


class WF(object):

    def __init__(self):
        self.phi = PHI()
        self.jast = JAST()

    def __call__(self, x, der=0, manual=False):

        if der == 0:
            return self.jast(x) * self.phi(x)

        if der == 1:

            jast = self.jast(x)
            djast = self.jast(x, der=1)

            phi = self.phi(x)
            dphi = self.phi(x, der=1)

            if not manual:

                djast_phi = djast.sum(0).unsqueeze(0) * phi
                dphi_jast = dphi * jast
                return djast_phi + dphi_jast

            else:
                out = torch.zeros(2, 2, 2)

                out[0, 0, 0] = djast[0, 0] * phi[0, 0] \
                    + jast[0]*dphi[0, 0]

                out[0, 0, 1] = djast[0, 1] * phi[0, 1] \
                    + jast[1]*dphi[0, 1]

                out[0, 1, 0] = djast[0, 0] * phi[1, 0]

                out[0, 1, 1] = djast[0, 1] * phi[1, 1]

                out[1, 0, 0] = djast[1, 0] * phi[0, 0]

                out[1, 0, 1] = djast[1, 1] * phi[0, 1]

                out[1, 1, 0] = djast[1, 0] * phi[1, 0] \
                    + jast[0] * dphi[1, 0]

                out[1, 1, 1] = djast[1, 1] * phi[1, 1] \
                    + jast[1] * dphi[1, 1]

                return out

        if der == 2:

            jast = self.jast(x)
            djast = self.jast(x, der=1)
            d2jast = self.jast(x, der=2)

            phi = self.phi(x)
            dphi = self.phi(x, der=1)
            d2phi = self.phi(x, der=2)

            jast_d2phi = d2phi * jast
            djast_dphi = djast * dphi
            d2jast_phi = d2jast.sum(0).unsqueeze(0) * phi

            return jast_d2phi + 2 * djast_dphi + d2jast_phi

    def grad_auto(self, x, size):
        mat = self(x, der=0)[:size, :size]
        det_mat = torch.det(mat)
        return grad(det_mat, x, grad_outputs=torch.ones_like(det_mat))[0].sum() / det_mat

    def grad_jacobi(self, x, size):

        mat = self(x, der=0)[:size, :size]
        imat = torch.inverse(mat)

        opgrad = self(x, der=1)[:size, :size]

        return torch.trace(imat @ opgrad)

    def hess_auto(self, x, size):
        mat = self(x, der=0)[:size, :size]
        det_mat = torch.det(mat)

        return hess(det_mat, x).sum() / det_mat

    def hess_jacobi(self, x, size):

        mat = self(x, der=0)[:size, :size]
        imat = torch.inverse(mat)
        ophess = self(x, der=2)[:size, :size]

        return torch.trace(imat @ ophess)

    def hess_2_manual(self, x):

        mat = self(x, 0)
        dmat = self(x, 1, True)
        d2mat = self(x, 2)

        out = d2mat[0, 0] * mat[1, 1] + d2mat[1, 1] * \
            mat[0, 0] + 2*(dmat[:, 0, 0]*dmat[:, 1, 1]).sum()
        out -= (d2mat[0, 1]*mat[1, 0] + d2mat[1, 0] *
                mat[0, 1] + 2*(dmat[:, 0, 1]*dmat[:, 1, 0]).sum())
        return out / torch.det(mat)


if __name__ == "__main__":

    wf = WF()

    x = torch.tensor([1., 2.])
    x.requires_grad = True

    grad_auto = wf.grad_auto(x, size=2)
    grad_jacobi = wf.grad_jacobi(x, size=2)

    print(grad_auto)
    print(grad_jacobi)

    hess_auto = wf.hess_auto(x, size=2)
    hess_jacobi = wf.hess_jacobi(x, size=2)
    hess_manual = wf.hess_2_manual(x)

    print(hess_manual)
    print(hess_auto)
    print(hess_jacobi)
