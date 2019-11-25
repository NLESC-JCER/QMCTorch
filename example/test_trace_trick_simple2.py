import sys
import torch
from torch.autograd import Variable, grad


def getA(xx):

    a=1.
    alpha = torch.tensor([-1.*a,-2.*a]) * xx**2
    A = xx*torch.exp(alpha)
    return A


def getd2A(xx):

    a=1.
    nabalxx = torch.ones(2,2)
    lapxx = torch.zeros(2,2)

    exp = torch.exp(torch.tensor([-1.*a,-2.*a]) * xx**2)
    nablaexp = torch.tensor([-1.*a,-2.*a])*2*xx * exp
    lapexp = 2*torch.tensor([-1.*a,-2.*a])*exp  +  torch.tensor([-1.*a,-2.*a])*2*xx * nablaexp

    d2A = lapxx*exp + 2*nabalxx*nablaexp + xx*lapexp
    return d2A

def second_der_autograd(xx):


    out = getA(xx)

    # compute the jacobian            
    z = Variable(torch.ones(out.shape))
    jacob = grad(out,xx,
                 grad_outputs=z,
                 allow_unused=True,
                 create_graph=True)[0]
    
    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)
    
    for idim in range(jacob.shape[1]):
        tmp = grad(jacob[:,idim],xx,
                  grad_outputs=z,
                  retain_graph=True,
                  only_inputs=True,
                  allow_unused=True)[0]    
        
        hess[:,idim] = tmp[:,idim]
    
    return hess


x = Variable(torch.tensor([-2.,1.]))
xx = Variable(x.repeat(2,1).transpose(0,1))
xx.requires_grad = True

A = getA(xx)
d2A = getd2A(xx)
d2A_auto = second_der_autograd(xx)


print(d2A.sum())
print(d2A_auto.sum())








    