import torch
from torch import nn
from torch.utils.data import Dataset

class QMCDataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,index):
        return self.data[index,:]

class QMCLoss(nn.Module):

    def __init__(self,wf,method='variance'):

        super(QMCLoss,self).__init__()
        self.wf = wf
        self.method = method

    def forward(self,vals,pos):

        if self.method == 'variance':
            loss = self.wf.variance(pos)

        elif self.method == 'energy':
            loss = self.wf.energy(pos)

        elif callable(self.method):
            loss = nn.MSELoss()
            target = torch.tensor(self.method(pos.detach().numpy()))
            return loss(vals,target)

        else:
            raise ValueError('method must be variance, energy or callable')

        return loss