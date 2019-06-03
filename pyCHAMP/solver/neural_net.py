import numpy as np 
import torch
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from functools import partial
from pyCHAMP.solver.solver_base import SOLVER_BASE

import matplotlib.pyplot as plt

from tqdm import tqdm
import time



class QMC_DataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,index):
        return self.data[index,:]

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
        pos = tensor.torch(self.sampler.generate(self.wf.pdf))
        pos.requires_grad = True
        return pos

    def sample_(self,export=10):
        niter = self.sampler.nstep/export
        self.sampler.nstep = export


    def train(self,nepoch):

        pos = self.sample()
        pos = torch.rand(self.sampler.nwalkers,3)
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


class NN4PYSCF(SOLVER_BASE):

    def __init__(self, wf=None, sampler=None, optimizer=None):

        SOLVER_BASE.__init__(self,wf,sampler,None)
        self.opt = optim.SGD(self.wf.parameters(),lr=0.005, momentum=0.9, weight_decay=0.001)
        self.batchsize = 100


    def sample(self,ntherm=10):

        t0 = time.time()
        pos = self.sampler.generate(self.wf.pdf,ntherm=ntherm)
        pos = torch.tensor(pos)
        pos = pos.view(-1,self.sampler.ndim*self.sampler.nelec)
        pos.requires_grad = True
        return pos.float()

    def observalbe(self,func,pos):
        obs = []
        for p in tqdm(pos):
            obs.append( func(p).data.numpy().tolist() )
        return obs

    def train(self,nepoch,pos=None,ntherm=0):

        if pos is None:
            pos = self.sample(ntherm=ntherm)

        dataset = QMC_DataSet(pos)
        dataloader = DataLoader(dataset,batch_size=self.batchsize)
        qmc_loss = QMCLoss(self.wf,method='variance')
        
        cumulative_loss = []
        for n in range(nepoch):
            print('\n === epoch %d' %n)

            cumulative_loss.append(0) 
            for data in dataloader:
                
                print("\n data ", data.shape)

                data = Variable(data).float()
                data.requires_grad = True
                t0 = time.time()
                out = self.wf(data)
                print("\t WF done in %f" %(time.time()-t0))

                t0 = time.time()
                loss = qmc_loss(out,data)
                cumulative_loss[n] += loss
                print("\t Loss (%f) done in %f" %(loss,time.time()-t0))
                self.wf = self.wf.train()

                self.opt.zero_grad()

                t0 = time.time()
                loss.backward()
                print("\t Backward done in %f" %(time.time()-t0))

                t0 = time.time()
                self.opt.step()
                print("\t opt done in %f" %(time.time()-t0))

                q,r = torch.qr(self.wf.layer_mo.weight.transpose(0,1))
                self.wf.layer_mo.weight.data = q.transpose(0,1)
                print(self.wf.layer_mo.weight)
                print(self.wf.layer_ci.weight)

            print('=== epoch %d loss %f \n' %(n,cumulative_loss[n]))

            if 1:
                pos = self.sample(ntherm=ntherm)
                dataloader.dataset.data = pos

        plt.plot(cumulative_loss)
        plt.show()


if __name__ == "__main__":

    from pyCHAMP.solver.vmc import VMC
    from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF, WaveNet
    from pyCHAMP.wavefunction.neural_pyscf_wf_base import NEURAL_PYSCF_WF
    from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS


    wf = NEURAL_PYSCF_WF(atom='H 0 1 0; H 0 0 1',
                         #basis='dzp',
                         basis='sto-3g',
                         active_space=(1,1))

    sampler = METROPOLIS(nwalkers=100, nstep=100, 
                         step_size = 3., nelec = wf.nelec, 
                         ndim = 3, domain = {'min':-5,'max':5})

    net = NN4PYSCF(wf=wf,sampler=sampler)

    pos = net.sample(ntherm=0)
    pos = pos.reshape(100,100,6)
    var = net.observalbe(net.wf.variance,pos)
    plt.plot(var)
    
    

    net.wf.layer_mo.weight.data = torch.eye(net.wf.nao).double()
    for param in net.wf.layer_ci.parameters():
        param.requires_grad = False


    pos = net.sample(ntherm=0)
    pos = pos.reshape(100,100,6)
    var_ = net.observalbe(net.wf.variance,pos)
    plt.plot(var_)
    plt.show()

    net.train(50,pos=pos,ntherm=-1)
