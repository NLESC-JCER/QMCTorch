import numpy as np 

import torch
from torch import nn
from torch.autograd import Variable, grad
import torch.optim as optim
from torch.utils.data import DataLoader

from pyCHAMP.solver.solver_base import SOLVER_BASE
from pyCHAMP.solver.torch_utils import QMCDataSet, QMCLoss

import matplotlib.pyplot as plt

from tqdm import tqdm
import time
            
class DeepQMC(SOLVER_BASE):

    def __init__(self, wf=None, sampler=None, optimizer=None):

        SOLVER_BASE.__init__(self,wf,sampler,None)
        #self.opt = optim.SGD(self.wf.parameters(),lr=0.05, momentum=0.9, weight_decay=0.001)        
        self.opt = optim.Adam(self.wf.parameters(),lr=0.005)
        self.batchsize = 100

    def sample(self,ntherm=10,with_tqdm=True,pos=None):

        t0 = time.time()
        pos = self.sampler.generate(self.wf.pdf,ntherm=ntherm,with_tqdm=with_tqdm,pos=pos)
        pos = torch.tensor(pos)
        pos = pos.view(-1,self.sampler.ndim*self.sampler.nelec)
        pos.requires_grad = True
        return pos.float()

    def observalbe(self,func,pos):
        obs = []
        for p in tqdm(pos):
            obs.append( func(p).data.numpy().tolist() )
        return obs

    def get_wf(self,x):
        vals = self.wf(x)
        return vals.detach().numpy().flatten()

    def plot_wf(self,grad=False,hist=False,sol=None):

        X = Variable(torch.linspace(-5,5,100).view(100,1,1))
        X.requires_grad = True

        vals = self.wf(X)
        vn = vals.detach().numpy().flatten()
        vn /= np.linalg.norm(vn)
        xn = X.detach().numpy().flatten()
        plt.plot(xn,vn)

        if grad:
            kin = self.wf.kinetic_autograd(X)
            g = np.gradient(vn,xn)
            h = np.gradient(g,xn)
            plt.plot(xn,kin.detach().numpy())
            plt.plot(xn,h)

        if hist:
            pos = self.sample(ntherm=-1)
            plt.hist(pos.detach().numpy(),density=False)

        if callable(sol):
            vs = sol(xn)
            plt.plot(xn,vs)
        
        plt.show()

    def train(self,nepoch,pos=None,ntherm=10,nresample=100,
              loss='variance',sol=None):
        '''Train the model.

        Arg:
            nepoch : number of epoch
            pos : presampled electronic poition
            ntherm : thermalization of the MC sampling
            nresample : number of MC step during the resampling
            loss : loss used ('energy','variance' or callable (for supervised)
            sol : anayltical solution for plotting (callable)
        '''

        if pos is None:
            pos = self.sample(ntherm=ntherm)

        dataset = QMCDataSet(pos)
        dataloader = DataLoader(dataset,batch_size=self.batchsize)
        qmc_loss = QMCLoss(self.wf,method=loss)
        
        XPLOT = Variable(torch.linspace(-5,5,100).view(100,1,1))
        xp = XPLOT.detach().numpy().flatten()
        vp = self.get_wf(XPLOT)
        
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot(xp,vp,color='red')
        if callable(sol):
            line2, = ax.plot(xp,sol(xp),color='blue')
        
        cumulative_loss = []
        for n in range(nepoch):

            cumulative_loss.append(0) 
            for data in dataloader:
                
                lpos = Variable(data).float()
                lpos.requires_grad = True
                vals = self.wf(lpos)

                loss = qmc_loss(vals,lpos)
                cumulative_loss[n] += loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            self.wf.fc.weight.data /= self.wf.fc.weight.data.norm() 

            vp = self.get_wf(XPLOT)
            line1.set_ydata(vp)
            fig.canvas.draw()            

            print('epoch %d loss %f' %(n,cumulative_loss[n]))
            print('variance : %f' %self.wf.variance(pos))
            print('energy : %f' %self.wf.energy(pos))

            self.sampler.nstep=nresample
            pos = self.sample(pos=pos.detach().numpy(),ntherm=ntherm,with_tqdm=False)
            dataloader.dataset.data = pos

        plt.plot(cumulative_loss)
        plt.show()






