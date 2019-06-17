import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF_BASE

from pyCHAMP.wavefunction.rbf import RBF
from pyCHAMP.solver.deepqmc import DeepQMC
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS


from pyCHAMP.solver.plot import plot_wf_3d
from pyCHAMP.solver.plot import plot_results_3d as plot_results

import matplotlib.pyplot as plt

import numpy as np


class RBF_H2plus(NEURAL_WF_BASE):

    def __init__(self,centers):
        super(RBF_H2plus,self).__init__(1,3)

        # get the RBF centers 
        self.centers = centers
        self.ncenter = len(self.centers)

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, self.ncenter, centers=self.centers, opt_centers=False)
        
        # define the fc layer
        self.fc = nn.Linear(self.ncenter, 1, bias=False)

        # initiaize the fc layer
        self.fc.weight.data.fill_(1.)
        self.fc.clip = False
        

    def forward(self,x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''

        batch_size = x.shape[0]
        x = self.rbf(x)
        x = self.fc(x)
        return x.view(-1,1)

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi

        TODO : vecorize that !! The solution below doesn't really wirk :(def plot_observable(obs_dict,e0=None,ax=None):)
        '''

        c0 = self.centers[0,:]
        c1 = self.centers[1,:]

        r0 = torch.sqrt(   ((pos-c0)**2).sum(1)  )
        r1 = torch.sqrt(   ((pos-c1)**2).sum(1)  )

        p0 = (-1./r0).view(-1,1)
        p1 = (-1./r1).view(-1,1)
        return p0+p1

        # r = torch.sqrt(((self.centers[:,None,:]-pos[None,...])**2).sum(2)).view(-1,self.ncenter)
        # return (-1./r).sum(1).view(-1,1)
   

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        return 0

def h2plus_sol(pos):
    '''Solution of the H2 + problem.'''

    centers = torch.tensor([[0.,0.,1.],[0.,0.,-1]])
    beta = 4.

    c0 = centers[0,:]
    c1 = centers[1,:]

    r0 = torch.sqrt(   ((pos-c0)**2).sum(1)  )
    r1 = torch.sqrt(   ((pos-c1)**2).sum(1)  )

    p0 = 2*torch.exp(-beta*r0).view(-1,1)
    p1 = 2*torch.exp(-beta*r1).view(-1,1)

    return p0+p1

    # r = torch.sqrt(((centers[:,None,:]-pos[None,...])**2).sum(2)).view(-1,2)
    # return 2*torch.exp(-1.0 * r).sum(1).view(-1,1)
    

# wavefunction
centers = torch.tensor([[0.,0.,1.5],[0.,0.,-1.5]])
wf = RBF_H2plus(centers)

#sampler
sampler = METROPOLIS(nwalkers=250, nstep=1000, 
                     step_size = 3., nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-5,'max':5})

# optimizer
opt = optim.Adam(wf.parameters(),lr=0.005)

# domain for the RBF Network
domain = {'xmin':-2.,'xmax':2.,'ymin':-2.,'ymax':2.,'zmin':-2.,'zmax':2.}
ncenter = [11,11,11]

# network
net = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)
pos = None
obs_dict = None

plot_wf_3d(net,domain,ncenter,sol=h2plus_sol,isoval=0.02,poly=False,hist=True)

# pos = Variable(net.sample())
# pos.requires_grad=True
# vals = net.wf(pos)


# optimize the position of the centers
net.wf.fc.weight.requires_grad = False
net.wf.rbf.centers.requires_grad = True

# train
pos,obs_dict = net.train(1000,
         batchsize=250,
         pos = pos,
         obs_dict = obs_dict,
         resample=100,
         ntherm=-1,
         loss = 'variance')

plot_results(net,obs_dict,domain,ncenter,isoval=0.02,sol=h2plus_sol,hist=True)






