import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF_BASE

from pyCHAMP.wavefunction.rbf import RBF_Slater_NELEC as RBF
from pyCHAMP.solver.deepqmc import DeepQMC
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS


from pyCHAMP.solver.mesh import regular_mesh_3d

from pyCHAMP.solver.plot import plot_wf_3d
from pyCHAMP.solver.plot import plot_results_3d as plot_results

import matplotlib.pyplot as plt

import numpy as np


class RBF_H(NEURAL_WF_BASE):

    def __init__(self,centers,sigma):
        super(RBF_H,self).__init__(1,3)

        # get the RBF centers 
        self.ncenter = len(centers)

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, 
                       self.ncenter, 
                       centers=centers,
                       nelec=self.nelec,
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
        r0 = torch.sqrt(   ((pos-c0)**2).sum(1)  ) 
        p0 = (-1./r0).view(-1,1)

        return p0 
   

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        return 0

    def nuclear_repulsion(self):
        return 0


def h_sol(pos):
    '''Solution of the H2 + problem.'''

    centers = torch.tensor([[0.,0.,0.]])
    beta = 1.
    c0 = centers[0,:]
    r0 = torch.sqrt(((pos-c0)**2).sum(1))
    p0 = 2. * torch.exp(-beta*r0).view(-1,1)
    return p0

def plot_wf(net,boundary = 5.,npts = 5):
    '''Plot the wavefunction.'''
    domain = {'xmin':-boundary,'xmax':boundary,
              'ymin':-boundary,'ymax':boundary,
              'zmin':-boundary,'zmax':boundary}

    points = regular_mesh_3d(xmin=domain['xmin'],xmax=domain['xmax'],
                            ymin=domain['ymin'],ymax=domain['ymax'],
                            zmin=domain['zmin'],zmax=domain['zmax'],
                            nx=npts,ny=npts,nz=npts)


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


def single_point(net,sigma=1.):
    ''' Compute a single point and plot the data.'''
    net.wf.rbf.sigma.data[:] = sigma

    pos = Variable(net.sample())
    pos.requires_grad = True

    e = net.wf.energy(pos)
    print('Energy   :', e)

    s = net.wf.variance(pos)
    print('Variance :', s)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    d = pos.detach().numpy()
    ax.scatter(d[:,0],d[:,1],d[:,2],alpha=0.01)

    plt.show()

def sigma_opt(net,sigma=1.5,loss='variance'):
    '''Optimize the sigma of the rbf.'''
    
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
    centers = torch.tensor([[0.,0.,0.]])
    sigma = torch.tensor([1.])
    wf = RBF_H(centers=centers,sigma=sigma)

    #sampler
    sampler = METROPOLIS(nwalkers=1000, nstep=1000,
                         step_size = 0.5, nelec = wf.nelec, 
                         ndim = wf.ndim, domain = {'min':-5,'max':5})

    # optimizer
    opt = optim.Adam(wf.parameters(),lr=0.005)
    #opt = optim.SGD(wf.parameters(),lr=0.1)

    # network
    net = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)
    
    # single point
    single_point(net,sigma=1.)















