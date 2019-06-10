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


class UnitNormClipper(object):

    def __call__(self,module):
        if hasattr(module,'weight'):
            w = module.weight.data
            w.div_(torch.norm(w).expand_as(w))

class ZeroOneClipper(object):
    
    def __call__(self, module):
        if hasattr(module,'weight'):
            w = module.weight.data
            w.sub_(torch.min(w)).div_(torch.norm(w).expand_as(w))
            
class DeepQMC(SOLVER_BASE):

    def __init__(self, wf=None, sampler=None, optimizer=None):

        SOLVER_BASE.__init__(self,wf,sampler,None)
        #self.opt = optim.SGD(self.wf.parameters(),lr=0.05, momentum=0.9, weight_decay=0.001)        
        #self.opt = optim.Adam(self.wf.parameters(),lr=0.005)
        self.opt = optimizer

    def sample(self,ntherm=-1,with_tqdm=True,pos=None):

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
        xn = X.detach().numpy().flatten()

        if callable(sol):
            vs = sol(xn)
            plt.plot(xn,vs,color='#b70000',linewidth=4,linestyle='--')

        vals = self.wf(X)
        vn = vals.detach().numpy().flatten()
        vn /= np.linalg.norm(vn)
        plt.plot(xn,vn,color='black',linewidth=2)

        if grad:
            kin = self.wf.kinetic_autograd(X)
            g = np.gradient(vn,xn)
            h = np.gradient(g,xn)
            plt.plot(xn,kin.detach().numpy())
            plt.plot(xn,h)

        if hist:
            pos = self.sample(ntherm=-1)
            plt.hist(pos.detach().numpy(),density=False)
        
        plt.grid()
        plt.show()

    def train(self,nepoch,
              batchsize=32,
              pos=None,
              ntherm=-1,
              resample=100,
              loss='variance',
              sol=None,
              fig=None):
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

        self.sampler.nstep=resample

        self.dataset = QMCDataSet(pos)
        self.dataloader = DataLoader(self.dataset,batch_size=batchsize)
        self.loss = QMCLoss(self.wf,method=loss)
        
        XPLOT = Variable(torch.linspace(-5,5,100).view(100,1,1))
        xp = XPLOT.detach().numpy().flatten()
        vp = self.get_wf(XPLOT)
        
        # plt.ion()
        # fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        line1, = ax.plot(xp,vp,color='red')
        line3, = ax.plot(self.wf.rbf.centers.detach().numpy(),self.wf.fc.weight.detach().numpy().T,'o')
        line4, = ax.plot(self.wf.rbf.centers.detach().numpy(),np.zeros(self.wf.ncenter),'X')
        if callable(sol):
            line2, = ax.plot(xp,sol(xp),color='blue')
        
        cumulative_loss = []
        #clipper = UnitNormClipper()
        clipper = ZeroOneClipper()
        obs_dict = {'energy':[],'variance':[],'loss':[],'local_energy':[]}

        for n in range(nepoch):

            cumulative_loss = 0
            for data in self.dataloader:
                
                lpos = Variable(data).float()
                lpos.requires_grad = True
                vals = self.wf(lpos)

                loss = self.loss(vals,lpos)
                cumulative_loss += loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.wf.fc.apply(clipper)

            vp = self.get_wf(XPLOT)
            line1.set_ydata(vp)

            line3.set_xdata(self.wf.rbf.centers.detach().numpy())
            line3.set_ydata(self.wf.fc.weight.detach().numpy().T)

            line4.set_xdata(self.wf.rbf.centers.detach().numpy())
            l4 = (self.wf.fc.weight.grad.detach().numpy().T)**2
            l4 /= np.linalg.norm(l4)
            line4.set_ydata(l4)

            fig.canvas.draw()            


            locale_ = self.wf.local_energy(pos)
            e_ = locale_.mean()
            v_ = locale_.var()

            print('epoch %d loss %f' %(n,cumulative_loss))
            print('variance : %f' %v_)
            print('energy : %f' %e_)

            obs_dict['energy'].append(e_.detach().numpy().tolist())
            obs_dict['variance'].append(v_.detach().numpy().tolist())
            obs_dict['local_energy'].append(locale_.detach().numpy().tolist())
            obs_dict['loss'].append(cumulative_loss.detach().numpy().tolist())

            if self.sampler.nstep > 0:
                pos = self.sample(pos=pos.detach().numpy(),ntherm=ntherm,with_tqdm=False)
                self.dataloader.dataset.data = pos

        return pos, obs_dict


    def plot_observable(self,obs_dict):

        n = len(obs_dict['energy'])
        epoch = np.arange(n)

        emax = [np.quantile(e,0.75) for e in obs_dict['local_energy'] ]
        emin = [np.quantile(e,0.25) for e in obs_dict['local_energy'] ]

        plt.fill_between(epoch,emin,emax,alpha=0.5,color='#4298f4')
        plt.plot(epoch,obs_dict['energy'],color='#144477')
        plt.grid()
        plt.xlabel('Number of epoch')
        plt.ylabel('Energy')
        plt.show()

    def plot_results(self,obs_dict,sol=None,e0=None):

        fig = plt.figure()
        ax0 = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)

        X = Variable(torch.linspace(-5,5,100).view(100,1,1))
        X.requires_grad = True
        xn = X.detach().numpy().flatten()

        if callable(sol):
            vs = sol(xn)
            ax0.plot(xn,vs,color='#b70000',linewidth=4,linestyle='--')

        vals = self.wf(X)
        vn = vals.detach().numpy().flatten()
        vn /= np.linalg.norm(vn)
        ax0.plot(xn,vn,color='black',linewidth=2)
        ax0.grid()
        ax0.set_xlabel('X')
        ax0.set_ylabel('Wavefuntion')


        n = len(obs_dict['energy'])
        epoch = np.arange(n)

        emax = [np.quantile(e,0.75) for e in obs_dict['local_energy'] ]
        emin = [np.quantile(e,0.25) for e in obs_dict['local_energy'] ]

        ax1.fill_between(epoch,emin,emax,alpha=0.5,color='#4298f4')
        ax1.plot(epoch,obs_dict['energy'],color='#144477')
        if e0 is not None:
            ax1.axhline(e0,color='black',linestyle='--')

        ax1.grid()
        ax1.set_xlabel('Number of epoch')
        ax1.set_ylabel('Energy')

        plt.show()
