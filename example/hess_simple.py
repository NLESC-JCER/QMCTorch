import torch
from torch.autograd import grad, Variable
from time import time


def f(xy):
    x, y = xy[:, 0], xy[:, 1]
    return x**2+y**3 - 4*x**3*y**2


def jac_f(xy):
    f = torch.zeros(xy.shape)
    x, y = xy[:, 0], xy[:, 1]

    f[:, 0] = 2*x - 12*x**2*y**2
    f[:, 1] = 3*y**2 - 8 * x**3*y
    return f


def hess_f(xy):
    f = torch.zeros(xy.shape)
    x, y = xy[:, 0], xy[:, 1]
    f[:, 0] = 2 - 24*x*y**2
    f[:, 1] = 6*y - 8*x**3
    return f


def hess_loop(out, pos):

    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos, grad_outputs=z, create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[1]):
        tmp = grad(jacob[:, idim], pos,
                   grad_outputs=z,
                   retain_graph=True,
                   create_graph=False,
                   allow_unused=False)[0]
        hess[:, idim] = tmp[:, idim]

    return jacob, hess


def hess_loop_fast(out, pos):

    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos, grad_outputs=z, create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[1]):
        z = Variable(torch.zeros(jacob.shape))
        z[:, idim] = 1.
        tmp = grad(jacob, pos,
                   grad_outputs=z,
                   retain_graph=True,
                   create_graph=False,
                   allow_unused=False)[0]
        hess[:, idim] = tmp[:, idim]
    return jacob, hess


def hess_fast(out, pos):

    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, pos, grad_outputs=z,
                 retain_graph=True, create_graph=True)[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[1]):
        z = Variable(torch.zeros(jacob.shape))
        z[:, idim] = 1.
        jacob.backward(z, retain_graph=True)
        print(pos.grad.data)
        hess[:, idim] = pos.grad.data[:, idim]

    return jacob, hess


Nw = 1000
Ndim = 2
xy = torch.rand(Nw, Ndim)
xy.requires_grad = True
out = f(xy)


jsol = jac_f(xy)
hsol = hess_f(xy)

t0 = time()
j2, h2 = hess_loop(out, xy)
print('loop : %f' % (time()-t0))
print((jsol-j2).norm())
print((hsol-h2).norm())

t0 = time()
j1, h1 = hess_fast(out, xy)
print('Fast : %f' % (time()-t0))

print((jsol-j1).norm())
print((hsol-h1).norm())
