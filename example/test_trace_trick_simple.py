import sys
import torch
from torch.autograd import Variable, grad


def btrace(M):
    return torch.diagonal(M,dim1=-2,dim2=-1).prod(-1)

def getA(x1,x2):
    a=1
    A = torch.exp(torch.tensor([[-a*x1,-2*a*x1],[-a*x2,-2*a*x2]]))
    A.requires_grad = True
    return A

def getA_(x):
    a=1.
    xx = x.repeat(2,1).transpose(0,1)
    xx = torch.tensor([-1.*a,-2.*a]) * xx
    return torch.exp(xx)


def getdA(x1,x2):
    a=1
    A = torch.exp(torch.tensor([[-a*x1,-2*a*x1],[-a*x2,-2*a*x2]]))
    dAdx1 = torch.tensor([[-a,-2*a],[0,0]]) * A
    dAdx2 = torch.tensor([[0,0],[-a,-2*a]]) * A
    return dAdx1,dAdx2



def getd2A(x1,x2):
    a=1
    A = torch.exp(torch.tensor([[-a*x1,-2*a*x1],[-a*x2,-2*a*x2]]))
    d2A1 = torch.tensor([[a**2,4*a**2],[0,0]]) * A
    d2A2 = torch.tensor([[0,0],[a**2,4*a**2]]) * A
    return d2A1,d2A2

def getdetA(x1,x2):
    a=1
    return torch.exp(-a*(x1+x2))*(torch.exp(-a*x2)-torch.exp(-a*x1)) 

def d2detA(x1,x2):
    a=1

    d2x1 = a**2*torch.exp(-a*(x1+x2))*(torch.exp(-a*x2)-2*torch.exp(-a*x1)) \
         -2 * a**2 * torch.exp(-a*(x1+x2)) * torch.exp(-a*x1)

    d2x2 = -a**2*torch.exp(-a*(x1+x2))*(torch.exp(-a*x1)-2*torch.exp(-a*x2)) \
         + 2 * a**2 * torch.exp(-a*(x1+x2)) * torch.exp(-a*x2)

    return d2x1,d2x2

def ddetA(x1,x2):
    a=1
    dx1 = -a*torch.exp(-a*(x1+x2))*(torch.exp(-a*x2)-2*torch.exp(-a*x1))
    dx2 = a*torch.exp(-a*(x1+x2))*(torch.exp(-a*x1)-2*torch.exp(-a*x2))
    return dx1,dx2

def first_der_autograd(x1,x2):


    xx = Variable(torch.tensor([x1,x2]))
    xx.requires_grad = True
    out = getA_(xx)
    print(out)
    out = torch.det(out)
    print(out)

    z = Variable(torch.ones(out.shape))
    jac = grad(out,xx,
             grad_outputs=z,
             only_inputs=True)[0]

    return jac

def second_der_autograd(x1,x2):

    xx = Variable(torch.tensor([x1,x2]))
    xx.requires_grad = True
    out = getA_(xx)
    out = torch.det(out)

    z = Variable(torch.ones(out.shape))
    jac = grad(out,xx,
             grad_outputs=z,
             only_inputs=True,
             create_graph=True)[0]

    h1 = grad(jac[0],xx,allow_unused=True,retain_graph=True)
    h2 = grad(jac[1],xx,allow_unused=True,retain_graph=True)
    
    return h1,h2,jac


x1 = Variable(torch.tensor([-1.]))
x2 = Variable(torch.tensor([1.]))

x1.requires_grad = True
x2.requires_grad = True

jac = first_der_autograd(x1,x2)

dD1, dD2 = ddetA(x1,x2)
A = getA(x1,x2)
invA  = torch.inverse(A)

detA = getdetA(x1,x2)
dA1,dA2 = getdA(x1,x2)

t1 = torch.trace(invA@dA1)
t2 = torch.trace(invA@dA2)

assert(torch.allclose(t1,dD1/detA))
assert(torch.allclose(t2,dD2/detA))

#assert(torch.allclose(t1,jac[0][0]/detA))
#assert(torch.allclose(t2,jac[0][0]/detA))


d2D1, d2D2 = d2detA(x1,x2)
d2A1,d2A2 = getd2A(x1,x2)

tt1 = torch.trace(invA@d2A1)
tt2 = torch.trace(invA@d2A2)

assert(torch.allclose(tt1,d2D1/detA))
assert(torch.allclose(tt2,d2D2/detA))


h1,h2,jjac = second_der_autograd(x1,x2)













    