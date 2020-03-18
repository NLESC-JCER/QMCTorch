import torch
from torch.optim import Optimizer
from tqdm import tqdm


class StochasticReconfiguration(Optimizer):

    def __init__(self, params, wf, lr=1E-2):
        if lr < 0.0:
            raise ValueError("Invalid value of tau :{}".format(tau))
        defaults = dict(lr=lr, wf=wf)
        super(StochasticReconfiguration, self).__init__(params, defaults)
        self.eloc = None
        self.lpos_needed = True

    def step(self, pos):
        """Performs a single optimization step

        Args:
            pos (torch.tensor) : used as global here ...
        """

        grad_e = self._collect_grad()
        S = self.get_overlap_matrix(pos)

        tau = self.defaults['lr']
        deltap, _ = torch.solve(-0.5*tau*grad_e.view(-1, 1), S)
        self.update_parameters(deltap)

    def get_overlap_matrix(self, pos):
        '''Get the overlap matrix
        Sij = <  psi_i / psi psi_j/psi > - <psi_i/psi><psi_j/psi>
        '''

        nbatch = pos.shape[0]

        ninp = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    ninp += p.numel()

        S = torch.zeros(ninp, ninp)
        sum_grads = torch.zeros(ninp)

        self.zero_grad()
        psi = self.defaults['wf'](pos)

        for ibatch in tqdm(range(nbatch)):

            psi[ibatch].backward(retain_graph=True)
            grads = self._collect_grad() / psi[ibatch]
            S += grads.reshape(-1, 1) @ grads.reshape(1, -1)
            sum_grads += grads
            self.zero_grad()

        sum_grads /= nbatch
        S -= sum_grads.reshape(-1, 1) @ sum_grads.reshape(1, -1)
        return S

    def _collect_grad(self, init=False):

        grad = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if init:
                        grad.append(torch.tensor([0.]))
                    else:
                        grad.append(p.grad.data.view(-1))
        return torch.cat(grad)

    def update_parameters(self, deltap):

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
