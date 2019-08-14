import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SOLVER_BASE(object):

    def __init__(self,wf=None, sampler=None, optimizer=None):

        self.wf = wf
        self.sampler = sampler
        self.optimizer = optimizer  

        self.history = {'eneregy':[],'variance':[],'param':[]}

        if optimizer is not None:
            self.optimizer.func = self.wf.energy
            self.optimizer.grad = self.wf.energy_gradient

    def sample(self,ntherm=-1,with_tqdm=True,pos=None):
        ''' sample the wave function.'''
        
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



    def energy(self,param,pos):
        return self.wf.energy(param,pos)

    def variance(self,param,pos):
        return self.wf.variance(param,pos)

    def single_point(self,param):
        pos = self.sample(param)
        e = self.energy(param,pos)
        s = self.variance(param,pos)
        return pos, e, s

    def plot_density(self,pos):

        if self.wf.ndim == 1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if self.wf.nelec == 1:
                plt.hist(pos)
            else:
                for ielec in range(self.wf.nelec):
                    plt.hist(pos[ielec,:])

        elif self.wf.ndim == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for ielec in range(self.wf.nelec):
                plt.scatter(pos[:,ielec*2],pos[:,ielec*2+1])

        elif self.wf.ndim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            for ielec in range(self.wf.nelec):
                ax.scatter(pos[:,ielec*3],pos[:,ielec*3+1],pos[:,ielec*3+2])
        plt.show()

    def plot_history(self):

        plt.plot(self.history['energy'])
        plt.plot(self.history['variance'])
        plt.show()
