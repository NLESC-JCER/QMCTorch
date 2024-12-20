import torch
from torch.autograd import grad
from torch.autograd.variable import Variable


def _hess(val, pos):
    """get the hessian of the jastrow values.
    of a given orbital terms
    Warning thos work only because the orbital term are dependent
    of a single rij term, i.e. fij = f(rij)

    Args:
        pos ([type]): [description]
    """
    print(pos.shape)
    print(val.shape)
    gval = grad(val, pos, grad_outputs=torch.ones_like(val), create_graph=True)[0]

    grad_out = Variable(torch.ones(*gval.shape[:-1]))
    hval = torch.zeros_like(gval)

    for idim in range(gval.shape[-1]):
        tmp = grad(
            gval[..., idim],
            pos,
            grad_outputs=grad_out,
            only_inputs=True,
            create_graph=True,
        )[0]
        hval[..., idim] = tmp[..., idim]

    return hval, gval


x = torch.rand(5, 4, 28, 3)
x.requires_grad = True
v = (x**3).prod(-1)

h, g = _hess(v, x)
