import torch
from torch.optim import Optimizer


class StochasticReconfiguration(Optimizer):

    def __init__(self, params, tau=1E-2):
        if tau < 0.0:
            raise ValueError("Invalid value of tau :{}".format(tau))
        defaults = dict(tau=tau)
        super(StochasticReconfiguration, self).__init__(params, defaults)

    def step(self, closure):
        """Performs a single optimization step

        Args:
            closure (callable): [description]
        """

        grad_e = self.get_gradient_var_energy()
        S = self.get_overlap_matrix(closure)
        tau = self.defaults['tau']
        deltap, _ = torch.solve(-0.5*tau*grad_e.view(-1, 1), S)

        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0

                numel = p.numel()
                p.data.add_(deltap[offset:offset + numel].view_as(p.data))
                offset += numel

    def get_overlap_matrix(self, closure):

        psi = closure()

        inp = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    inp.append(p)

        for ibatch in range(psi.shape[0]):
            grads = torch.autograd.grad(psi[ibatch], inp, retain_graph=True)
            grads = torch.cat([g.view(-1)/psi[ibatch] for g in grads])
            if ibatch == 0:
                S = grads * grads.view(-1, 1)
            else:
                S += grads * grads.view(-1, 1)

        return S / psi.shape[0]

    def get_gradient_var_energy(self):

        g = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g.append(p.grad.data.view(-1))
        return torch.cat(g)
