import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF_BASE

from pyCHAMP.wavefunction.rbf import RBF
from pyCHAMP.solver.deepqmc import DeepQMC
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS
from pyCHAMP.sampler.pymc3 import PYMC3_TORCH as PYMC3

from pyCHAMP.solver.mesh import regular_mesh_3d

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
        self.rbf = RBF(self.ndim_tot, 
                       self.ncenter, 
                       centers=self.centers, 
                       opt_centers=False,
                       kernel='gaussian',
                       sigma=1.)
        
        # define the fc layer
        self.fc = nn.Linear(self.ncenter, 1, bias=False)

        # initiaize the fc layer
        self.fc.weight.data.fill_(2.)
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

        r0 = torch.sqrt(   ((pos-c0)**2).sum(1)  ) + 1E-1
        r1 = torch.sqrt(   ((pos-c1)**2).sum(1)  ) + 1E-1

        p0 = (-1./r0).view(-1,1)
        p1 = (-1./r1).view(-1,1)
        

        return p0 + p1

        # r = torch.sqrt(((self.centers[:,None,:]-pos[None,...])**2).sum(2)).view(-1,self.ncenter)
        # return (-1./r).sum(1).view(-1,1)
   

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        return 0

    def nuclear_repulsion(self):
        c0 = self.centers[0,:]
        c1 = self.centers[1,:]
        rnn = torch.sqrt(   ((c0-c1)**2).sum()  )
        return (1./rnn).view(-1,1)

    def atomic_distance(self,pos=None):
        c0 = self.centers[0,:]
        c1 = self.centers[1,:]
        return torch.sqrt(   ((c0-c1)**2).sum()  )

def h2plus_sol(pos):
    '''Solution of the H2 + problem.'''

    centers = torch.tensor([[0.,0.,2.],[0.,0.,-2]])
    beta = 1.

    k = torch.sqrt( torch.tensor( (2*np.pi*beta)**3 ) )
    
    c0 = centers[0,:]
    c1 = centers[1,:]

    r0 = ((pos-c0)**2).sum(1)
    r1 = ((pos-c1)**2).sum(1)

    p0 = 2./k * torch.exp(-0.5*beta*r0).view(-1,1)
    p1 = 2./k * torch.exp(-0.5*beta*r1).view(-1,1)

    return p0+p1

    # r = torch.sqrt(((centers[:,None,:]-pos[None,...])**2).sum(2)).view(-1,2)
    # return 2*torch.exp(-1.0 * r).sum(1).view(-1,1)
    

# wavefunction
X = 0.5
centers = torch.tensor([[X,X,X],
                        [-X,-X,-X]])
wf = RBF_H2plus(centers)

#sampler
sampler = METROPOLIS(nwalkers=500, nstep=5000,
                     step_size = 0.5, nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-5,'max':5})

# optimizer
opt = optim.Adam(wf.parameters(),lr=0.005)
#opt = optim.SGD(wf.parameters(),lr=0.1)

# domain for the RBF Network
boundary = 5.
domain = {'xmin':-boundary,'xmax':boundary,
          'ymin':-boundary,'ymax':boundary,
          'zmin':-boundary,'zmax':boundary}
ncenter = [11,11,11]

# network
net = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)
obs_dict = {'local_energy':[],
            'atomic_distance':[]}

if 1:
    plot_wf_3d(net,domain,ncenter,sol=None,
               wf=True, isoval=0.01,
               hist=True,
               pot=False,pot_isoval=-1,
               grad=False, grad_isoval = 0.01)

if 0:
    X = np.linspace(0.1,2,25)
    energy, var = [], []
    for x in X:

        net.wf.rbf.centers.data[0,2] = -x
        net.wf.rbf.centers.data[1,2] = x
        pos = Variable(net.sample())
        pos.requires_grad = True
        e = net.wf.energy(pos)
        s = net.wf.variance(pos)

        energy.append(e)
        var.append(s)

    plt.plot(X,energy)
    plt.show()
    plt.plot(X,var)
    plt.show()
    

if 0:
    pos = Variable(net.sample())
    pos.requires_grad = True
    e = net.wf.energy(pos)
    s = net.wf.variance(pos)

    print('Energy   :', e)
    print('Variance :', s)
    sys.exit()



if 0:

    # do not optimize the weights of fc
    net.wf.fc.weight.requires_grad = False

    # optimize the position of the centers
    # do not optimize the std of the gaussian
    net.wf.rbf.sigma.requires_grad = False
    net.wf.rbf.centers.requires_grad = True

    # train
    pos,obs_dict = net.train(500,
             batchsize=500,
             pos = None,
             obs_dict = obs_dict,
             resample=5000,
             resample_every=25,
             ntherm=-1,
             loss = 'energy')

    plot_results(net,obs_dict,domain,ncenter,isoval=0.02,hist=True)






