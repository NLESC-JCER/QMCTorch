import numpy as np 
import torch
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from functools import partial
from pyCHAMP.solver.solver_base import SOLVER_BASE

import matplotlib.pyplot as plt

class QMC_DataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self,index):
        return self.data[:,index]

class QMCLoss(nn.Module):

    def __init__(self,wf,method='energy'):

        super(QMCLoss,self).__init__()
        self.wf = wf
        self.method = method

    def forward(self,out,pos):

        if self.method == 'variance':
            loss = self.wf.variance(pos)

        elif self.method == 'energy':
            loss = self.wf.energy(pos)

        return loss
            

class NN(SOLVER_BASE):

    def __init__(self, wf=None, sampler=None, optimizer=None):

        SOLVER_BASE.__init__(self,wf,sampler,None)
        self.opt = optim.SGD(self.wf.model.parameters(),lr=0.005, momentum=0.9, weight_decay=0.001)
        self.batchsize = 32


    def sample(self):
        pos = self.sampler.generate(self.wf.pdf)
        return pos

    def train(self,nepoch):

        pos = self.sample()
        pos = torch.rand(3,self.sampler.nwalkers)
        dataset = QMC_DataSet(pos)

        dataloader = DataLoader(dataset,batch_size=self.batchsize)
        qmc_loss = QMCLoss(self.wf,method='variance')
        
        cumulative_loss = []
        for n in range(nepoch):

            cumulative_loss.append(0) 
            for data in dataloader:
                
                data = Variable(data).float()
                out = self.wf.model(data)
                
                self.wf.model = self.wf.model.eval()
                loss = qmc_loss(out,data)
                cumulative_loss[n] += loss
                self.wf.model = self.wf.model.train()

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            print('epoch %d loss %f' %(n,cumulative_loss[n]))
            pos = self.sample()
            dataloader.dataset.data = pos.T

        plt.plot(cumulative_loss)
        plt.show()

if __name__ == "__main__":

    from pyCHAMP.solver.vmc import VMC
    from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF, WaveNet
    from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS


    class HarmOsc3D(NEURAL_WF):

        def __init__(self,model,nelec,ndim):
            NEURAL_WF.__init__(self, model, nelec, ndim)

        def nuclear_potential(self,pos):
            return torch.sum(0.5*pos**2,1)

        def electronic_potential(self,pos):
            return 0

    wf = HarmOsc3D(model=WaveNet,nelec=1, ndim=3)
    sampler = METROPOLIS(nwalkers=250, nstep=100, 
                         step_size = 3, nelec=1, 
                         ndim=3, domain = {'min':-2,'max':2})

    nn = NN(wf=wf,sampler=sampler)
    nn.train(100)
    #pos = torch.tensor(nn.sample()).float()
    #x = nn.wf.model(pos)