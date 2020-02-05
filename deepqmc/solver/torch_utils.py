import torch
from torch import nn
from torch.utils.data import Dataset


def set_torch_double_precision():
    torch.set_default_dtype = torch.float64
    torch.set_default_tensor_type(torch.DoubleTensor)


def set_torch_single_precision():
    torch.set_default_dtype = torch.float32
    torch.set_default_tensor_type(torch.FloatTensor)


class DataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :]


class Loss(nn.Module):

    def __init__(self, wf, method='variance', clip=False):

        super(Loss, self).__init__()
        self.wf = wf
        self.method = method
        self.clip = clip

        self.use_weight = 'weighted' in self.method
        if self.use_weight:
            self.weight = {'psi': None, 'psi0': None}

    def forward(self, pos):

        local_energies = self.wf.local_energy(pos)

        if self.clip:
            median = torch.median(local_energies)
            std = torch.std(local_energies)
            mask = (local_energies < median +
                    5*std) & (local_energies > median-5*std)
        else:
            mask = torch.ones_like(local_energies).type(torch.bool)

        if not self.use_weight:

            if self.method == 'variance':
                loss = torch.var(local_energies[mask])

            elif self.method == 'energy':
                loss = torch.mean(local_energies[mask])

            else:
                raise ValueError(
                    'method must be variance, energy, weighted-variance or weighted_energy')

        else:

            self.weight['psi'] = self.wf(pos)

            if self.weight['psi0'] is None:
                self.weight['psi0'] = self.weight['psi'].detach().clone()

            w = (self.weight['psi']/self.weight['psi0'])**2
            w /= w.sum()

            if self.method == 'weighted-variance':
                mu = torch.mean(local_energies)
                weighted_local_energies = (local_energies-mu)**2 * w
                loss = torch.mean(weighted_local_energies[mask])

            elif self.method == 'weighted-energy':
                weighted_local_energies = local_energies * w
                loss = torch.mean(weighted_local_energies[mask])

            else:
                raise ValueError(
                    'method must be variance, energy, weighted-variance or weighted_energy')

        return loss, local_energies


class OrthoReg(nn.Module):
    '''add a penalty to make matrice orthgonal.'''

    def __init__(self, alpha=0.1):
        super(OrthoReg, self).__init__()
        self.alpha = alpha

    def forward(self, W):
        ''' Return the loss : |W x W^T - I|.'''
        return self.alpha * torch.norm(W.mm(W.transpose(0, 1)) - torch.eye(W.shape[0]))


class UnitNormClipper(object):

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.div_(torch.norm(w).expand_as(w))


class ZeroOneClipper(object):

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.sub_(torch.min(w)).div_(torch.norm(w).expand_as(w))
