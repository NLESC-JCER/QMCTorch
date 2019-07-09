import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF_BASE

from pyCHAMP.wavefunction.rbf import RBF_Slater as RBF
from pyCHAMP.solver.deepqmc import DeepQMC
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS
from pyCHAMP.sampler.hamiltonian import HAMILTONIAN_TORCH as HAMILTONIAN
from pyCHAMP.sampler.pymc3 import PYMC3_TORCH as PYMC3

from pyCHAMP.solver.mesh import regular_mesh_3d

from pyCHAMP.solver.plot import plot_wf_3d
from pyCHAMP.solver.plot import plot_results_3d as plot_results

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import numpy as np


class RBF_H2plus(NEURAL_WF_BASE):

    def __init__(self,centers,sigma):
        super(RBF_H2plus,self).__init__(1,3)

        #self.centers = centers
        self.ncenter = len(centers)

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, 
                       self.ncenter, 
                       centers=centers, 
                       sigma=sigma)

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

        c0 = self.rbf.centers[0,:]
        c1 = self.rbf.centers[1,:]

        r0 = torch.sqrt(   ((pos-c0)**2).sum(1)  ) + 1E-3
        r1 = torch.sqrt(   ((pos-c1)**2).sum(1)  ) + 1E-3

        p0 = (-1./r0).view(-1,1)
        p1 = (-1./r1).view(-1,1)
        
        return p0 + p1

   

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        return 0

    def nuclear_repulsion(self):
        '''Compute the nuclear repusion.'''
        c0 = self.rbf.centers[0,:]
        c1 = self.rbf.centers[1,:]
        rnn = torch.sqrt(   ((c0-c1)**2).sum()  )
        return (1./rnn).view(-1,1)

    def atomic_distance(self,pos=None):
        '''get the distance between the atoms.'''
        c0 = self.rbf.centers[0,:]
        c1 = self.rbf.centers[1,:]
        return torch.sqrt(   ((c0-c1)**2).sum()  )

    def get_sigma(self,pos=None):
        '''get the sigma value.'''
        return self.rbf.sigma.data[0]

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

def plot_wf(net,boundary = 5.):
    '''Plot the wavefunction.'''
    domain = {'xmin':-boundary,'xmax':boundary,
              'ymin':-boundary,'ymax':boundary,
              'zmin':-boundary,'zmax':boundary}
    ncenter = [11,11,11]

    plot_wf_3d(net,domain,ncenter,sol=None,
           wf=True, isoval=0.01,
           hist=True,
           pot=False,pot_isoval=-1,
           grad=False, grad_isoval = 0.01)
    

def sigma_curve(net,S=np.linspace(0.5,1.5,25)):
    '''Compute and plot E(sigma) abd V(sigma).'''
    energy, var = [], []
    for x in S:

        net.wf.rbf.sigma.data[:] = x
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

def pos_curve(net,X=np.linspace(0.5,1.5,25)):
    '''Compute and plot E(R_hh) abd V(R_hh).'''
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


def single_point(net,x=1.5,loss='variance'):
    ''' Compute a single point and plot the data.'''
    net.wf.rbf.centers.data[0,2] = -x
    net.wf.rbf.centers.data[1,2] = x
    net.wf.rbf.centers.requires_grad = True

    pos = Variable(net.sample())
    pos.requires_grad = True

    e = net.wf.energy(pos)
    print('Energy   :', e)
    if loss=='energy':
        e.backward()

    s = net.wf.variance(pos)
    print('Variance :', s)
    if loss == 'variance':
        s.backward()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    d = pos.detach().numpy()
    ax.scatter(d[:,0],d[:,1],d[:,2],alpha=0.01)

    for i in range(2):
        x,y,z = net.wf.rbf.centers.data[i,:].numpy()
        u,v,w = net.wf.rbf.centers.grad.data[i,:].numpy()
        ax.quiver(x,y,z,-u,-v,-w,color='black',length=1.,normalize=True,pivot='middle')

    plt.show()

def geo_opt(net,x=1.25,sigma=1.20,loss='energy'):
    '''Optimize the gemoentry of the mol.'''
    net.wf.rbf.centers.data[0,2] = -x
    net.wf.rbf.centers.data[1,2] = x

    net.wf.rbf.sigma.data[:] = sigma

    # do not optimize the weights of fc
    net.wf.fc.weight.requires_grad = False

    # optimize the position of the centers
    # do not optimize the std of the gaussian
    net.wf.rbf.centers.requires_grad = True
    net.wf.rbf.sigma.requires_grad = False

    # variable
    obs_dict = {'local_energy':[],
                'atomic_distance':[],
                'get_sigma':[]}

    # train
    pos,obs_dict = net.train(250,
             batchsize=5000,
             pos = None,
             obs_dict = obs_dict,
             resample=250,
             resample_every=1,
             resample_from_last = True,
             ntherm=-1,
             loss = loss)

    boundary = 5.
    domain = {'xmin':-boundary,'xmax':boundary,
              'ymin':-boundary,'ymax':boundary,
              'zmin':-boundary,'zmax':boundary}
    ncenter = [11,11,11]

    plot_results(net,obs_dict,domain,ncenter,isoval=0.02,hist=True)

def sigma_opt(net,x=0.97,sigma=1.,loss='variance'):
    '''Optimize the sigma of the rbf.'''
    
    net.wf.rbf.centers.data[0,2] = -x
    net.wf.rbf.centers.data[1,2] = x

    net.wf.rbf.sigma.data[:] = sigma

    # do not optimize the weights of fc
    net.wf.fc.weight.requires_grad = False

    # # optimize the position of the centers
    # do not optimize the std of the gaussian
    net.wf.rbf.centers.requires_grad = False
    net.wf.rbf.sigma.requires_grad = True

    # variable
    obs_dict = {'local_energy':[],
                'atomic_distance':[],
                'get_sigma':[]}

    # train
    pos,obs_dict = net.train(500,
             batchsize=1000,
             pos = None,
             obs_dict = obs_dict,
             resample=100,
             resample_every=1,
             resample_from_last = True,
             ntherm=-1,
             loss = loss)

    boundary = 5.
    domain = {'xmin':-boundary,'xmax':boundary,
              'ymin':-boundary,'ymax':boundary,
              'zmin':-boundary,'zmax':boundary}
    ncenter = [11,11,11]
    plot_results(net,obs_dict,domain,ncenter,isoval=0.02,hist=True)


if __name__ == "__main__":

    # wavefunction
    X = 0.97 # <- opt ditance +0.97 and -0.97
    S = 1.20 # <- roughly ideal zeta parameter

    # define the RBF WF
    centers = torch.tensor([[0.,0.,-X],[0.,0.,X]])
    sigma = torch.tensor([S,S])
    wf = RBF_H2plus(centers=centers,sigma=sigma)

    #sampler
    sampler = METROPOLIS(nwalkers=2000, nstep=5000,
                         step_size = 0.1, nelec = wf.nelec, 
                         ndim = wf.ndim, domain = {'min':-5,'max':5})

    sampler_ham = HAMILTONIAN(nwalkers=250, nstep=250, 
                         step_size = 0.01, nelec = wf.nelec, 
                         ndim = wf.ndim, domain = {'min':-5,'max':5}, L=15)

    # optimizer
    opt = optim.Adam(wf.parameters(),lr=0.002) #,betas=(0.,0.99999))
    #opt = optim.SGD(wf.parameters(),lr=0.01)

    # network
    net = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)

    # compute a single point
    single_point(net,x=1.5,loss='variance')

    # optimize the gemoetry
    geo_opt(net,x=1.25,sigma=1.20,loss='variance')














