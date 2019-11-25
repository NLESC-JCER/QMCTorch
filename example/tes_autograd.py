import torch
from torch.autograd import grad, Variable

x = 3.14*torch.ones(2,2)
x.requires_grad = True

y = x**3

jac = grad(y,x,grad_outputs=torch.ones(y.shape))[0]

jac.backward(torch.ones(2,2),retain_graph=True)
print(x.grad.data)