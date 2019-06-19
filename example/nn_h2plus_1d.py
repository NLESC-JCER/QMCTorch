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

from pyCHAMP.solver.mesh import regular_mesh_3d

from pyCHAMP.solver.plot import plot_wf_1d, plotter1d
from pyCHAMP.solver.plot import plot_results_1d as plot_results

import matplotlib.pyplot as plt

import numpy as np


class RBF_H2plus(NEURAL_WF_BASE):

    def __init__(self,centers):
        super(RBF_H2plus,self).__init__(1,1)

        # get the RBF centers 
        self.centers = centers
        self.ncenter = len(self.centers)

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, 
                       self.ncenter, 
                       centers=self.centers, 
                       opt_centers=False,
                       kernel='slater',
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

        c0 = self.centers[0]
        c1 = self.centers[1]

        r0 = torch.sqrt(   ((pos-c0)**2)  ) + 1E-1
        r1 = torch.sqrt(   ((pos-c1)**2)  ) + 1E-1

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
        c0 = self.centers[0]
        c1 = self.centers[1]
        rnn = torch.sqrt(   ((c0-c1)**2).sum()  )
        return (1./rnn).view(-1,1)

    def atomic_distance(self,pos=None):
        c0 = self.centers[0]
        c1 = self.centers[1]
        return torch.sqrt(   ((c0-c1)**2).sum()  )

def h2plus_sol(pos):
    '''Solution of the H2 + problem.'''

    centers = torch.tensor([-0.75,0.75])
    beta = 1.

    c0 = centers[0]
    c1 = centers[1]

    r0 = torch.sqrt(((pos-c0)**2))
    r1 = torch.sqrt(((pos-c1)**2))

    p0 = 2.* torch.exp(-0.5*beta*r0).view(-1,1)
    p1 = 2.* torch.exp(-0.5*beta*r1).view(-1,1)

    return p0+p1
    

# wavefunction
X = 0.1
centers = torch.tensor([[X],
                        [-X]])
wf = RBF_H2plus(centers)

#sampler
sampler = METROPOLIS(nwalkers=1000, nstep=1000, 
                     step_size = 3., nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-5,'max':5})

# optimizer
opt = optim.Adam(wf.parameters(),lr=0.05)
#opt = optim.SGD(wf.parameters(),lr=0.1)

# domain for the RBF Network
boundary = 5.
domain = {'xmin':-boundary,'xmax':boundary}
ncenter = 51

# network
net = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)

obs_dict = {'local_energy':[],
            'atomic_distance':[]}

if 0:
    plot_wf_1d(net,domain,ncenter,sol=h2plus_sol,
               hist=False,
               pot=False,
               grad=True)

if 0:

    X = np.linspace(0.1,2,25)
    energy, var = [], []
    for x in X:

        net.wf.rbf.centers.data[0] = -x
        net.wf.rbf.centers.data[1] = x
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



if 1:


    plot1D = plotter1d(wf,domain,ncenter,sol=h2plus_sol)

    # do not optimize the weights of fc
    net.wf.fc.weight.requires_grad = False

    # optimize the position of the centers
    # do not optimize the std of the gaussian
    net.wf.rbf.sigma.requires_grad = False
    net.wf.rbf.centers.requires_grad = True

    # train
    pos,obs_dict = net.train(250,
             batchsize=500,
             pos = None,
             obs_dict = obs_dict,
             resample=200,
             ntherm=-1,
             loss = 'energy',
             plot=plot1D)

    plot_results(net,obs_dict,domain,ncenter)






