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

from pyCHAMP.wavefunction.wave_modules import SlaterPooling, TwoBodyJastrowFactor, ElectronDistance

from pyCHAMP.solver.plot import plot_wf_3d
from pyCHAMP.solver.plot import plot_results_3d as plot_results

import matplotlib.pyplot as plt

import numpy as np


class RBF_H2(NEURAL_WF_BASE):

    def __init__(self,centers,sigma):
        super(RBF_H2,self).__init__(2,3)

        # basis function
        self.centers = centers
        self.ncenters = len(centers)

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, 
                       self.ncenters, 
                       centers=centers, 
                       sigma=sigma,
                       nelec=self.nelec)
        
        # define the mo layer
        self.mo = nn.Linear(self.ncenters, self.ncenters, bias=False)
        mo_coeff =  torch.sqrt(torch.tensor([1./2.]))  * torch.tensor([[1.,1.],[1.,-1.]])
        self.mo.weight = nn.Parameter(mo_coeff.transpose(0,1))

        # jastrow
        self.edist = ElectronDistance(2,3)
        self.jastrow = TwoBodyJastrowFactor(1,1)

        # define the SD pooling layer
        self.configs = (torch.LongTensor([np.array([0])]), torch.LongTensor([np.array([0])]))
        self.nci = 1
        self.pool = SlaterPooling(self.configs,1,1)

        # define the linear layer
        self.fc = nn.Linear(self.nci, 1, bias=False)
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
        
        edist  = self.edist(x)
        J = self.jastrow(edist)

        x = self.rbf(x)
        x = self.mo(x)
        #x = self.pool(x) #<- issue with batch determinant
        x = (x[:,0,0]*x[:,1,0]).view(-1,1)
        return J*x

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi

        TODO : vecorize that !! The solution below doesn't really wirk :(def plot_observable(obs_dict,e0=None,ax=None):)
        '''
        
        p = torch.zeros(pos.shape[0])
        for ielec in range(self.nelec):
            pelec = pos[:,(ielec*self.ndim):(ielec+1)*self.ndim]
            for iatom in range(len(self.centers)):
                patom = self.rbf.centers[iatom,:]

                r = torch.sqrt(   ((pelec-patom)**2).sum(1)  ) + 1E-6
                p += (-1./r)

        return p.view(-1,1)

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''

        pot = torch.zeros(pos.shape[0])
        
        for ielec1 in range(self.nelec-1):
            epos1 = pos[:,ielec1*self.ndim:(ielec1+1)*self.ndim]
            
            for ielec2 in range(ielec1+1,self.nelec):
                epos2 = pos[:,ielec2*self.ndim:(ielec2+1)*self.ndim]
                
                r = torch.sqrt( ((epos1-epos2)**2).sum(1) ) + 1E-6
                pot = (1./r) 

        return pot.view(-1,1)

    def nuclear_repulsion(self):
        c0 = self.rbf.centers[0,:]
        c1 = self.rbf.centers[1,:]
        rnn = torch.sqrt(   ((c0-c1)**2).sum()  )
        return (1./rnn).view(-1,1)

    def atomic_distance(self,pos=None):
        c0 = self.rbf.centers[0,:]
        c1 = self.rbf.centers[1,:]
        return torch.sqrt(   ((c0-c1)**2).sum()  )

    def get_sigma(self,pos=None):
        return self.rbf.sigma.data[0]

    def get_mos(self,pos=None):
        return self.mo.weight.data


def plot_wf(net):
    '''Plot the density.'''
    pos = net.sample()
    net.plot_density(pos.detach().numpy())

def sigma_curve(net,X = np.linspace(0.5,2,11)):
    '''Plot the E(sigma), V(sigma) curves.'''
    energy, var = [], []
    for x in X:

        net.wf.rbf.sigma.data[:] = x
        pos = Variable(net.sample())
        pos.requires_grad = True
        e = net.wf.energy(pos)
        s = net.wf.variance(pos)

        energy.append(e.data.item())
        var.append(s.data.item())

    plt.plot(X,energy)
    plt.show()
    plt.plot(X,var)
    plt.show()

def pos_curve(net,X = np.linspace(0.2,1.5,11)):
    '''Plot the E(R_hh) and V(R_hh) cruves.'''

    energy, var = [], []
    K,Vnn,Ven,Vee = [],[],[],[]

    for x in X:

        net.wf.rbf.centers.data[0,2] = -x
        net.wf.rbf.centers.data[1,2] = x
        pos = Variable(net.sample())
        pos.requires_grad = True

        K.append(net.wf.kinetic_energy(pos).mean().data.item())
        Vnn.append(net.wf.nuclear_repulsion().data.item())
        Ven.append(net.wf.nuclear_potential(pos).mean().data.item())
        Vee.append(net.wf.electronic_potential(pos).mean().data.item())

        energy.append(net.wf.energy(pos).data.item())
        var.append(net.wf.variance(pos).data.item())

    plt.plot(X,energy,linewidth=4,c='black',label='Energy')
    plt.plot(X,K,label='K')
    plt.plot(X,Vee,label='Vee')
    plt.plot(X,Ven,label='Ven')
    plt.plot(X,Vnn,label='Vnn')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(X,var)
    plt.show()

def single_point(net,x=0.69,loss='variance',alpha=0.01):

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
    ax.scatter(d[:,0],d[:,1],d[:,2],alpha=alpha)
    ax.scatter(d[:,3],d[:,4],d[:,5],alpha=alpha, color='red')

    for i in range(2):
        x,y,z = net.wf.rbf.centers.data[i,:].numpy()
        u,v,w = net.wf.rbf.centers.grad.data[i,:].numpy()
        ax.quiver(x,y,z,-u,-v,-w,color='black',length=1.,normalize=True,pivot='middle')

    plt.show()


def geo_opt(net,x=1.25,sigma=1.20,loss='energy'):
    '''Optimize the geometry of the mol.'''

    # fix the mo coefficients
    mo_coeff =  torch.sqrt(torch.tensor([1./2.]))  * torch.tensor([[1.,1.],[1.,-1.]])
    net.wf.mo.weight = nn.Parameter(mo_coeff.transpose(0,1))
    net.wf.mo.weight.requires_grad = False

    # set the positions and sigmas
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
    pos,obs_dict = net.train(100,
             batchsize=1000,
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

def mo_opt(net,x=0.69,sigma=1.20,loss='variance'):
    '''Optimize the MO coefficients.'''

    # Change the mo coeffs
    q,_ = np.linalg.qr(np.array([[0.8,0.2],[0.2,-0.8]]))
    mo_coeff = torch.tensor(q).float()
    net.wf.mo.weight = nn.Parameter(mo_coeff.transpose(0,1))

    # set the other params
    net.wf.rbf.centers.data[0,2] = -x
    net.wf.rbf.centers.data[1,2] = x
    net.wf.rbf.sigma.data[:] = s 

    # optimize the weights of mo
    net.wf.mo.weight.requires_grad = True

    # do not optimize the position of the centers
    # do not optimize the std of the gaussian
    net.wf.rbf.centers.requires_grad = False
    net.wf.rbf.sigma.requires_grad = False

    # track obs
    obs_dict = {'local_energy':[],
            'atomic_distance':[],
            'get_sigma':[],
            'get_mos':[]}

    # train
    pos,obs_dict = net.train(150,
             batchsize=500,
             pos = None,
             obs_dict = obs_dict,
             resample=100,
             resample_from_last=True,
             resample_every=1,
             ntherm=-1,
             loss = loss)

    # domain for the RBF Network
    boundary = 5.
    domain = {'xmin':-boundary,'xmax':boundary,
              'ymin':-boundary,'ymax':boundary,
              'zmin':-boundary,'zmax':boundary}
    ncenter = [11,11,11]

    # plot the result
    plot_results(net,obs_dict,domain,ncenter,hist=True)



if __name__ == "__main__":

    # wavefunction 
    # bond distance : 0.74 A -> 1.38 a
    # ground state energy : -31.688 eV -> -1.16 hartree
    # bond dissociation energy 4.478 eV -> 0.16 hartree
    X = 0.69 # <- opt ditance +0.69 and -0.69
    S = 1.24 # <- roughly ideal zeta parameter


    # define the RBF WF
    centers = torch.tensor([[0.,0.,-X],[0.,0.,X]])
    sigma = torch.tensor([S,S])
    wf = RBF_H2(centers=centers,sigma=sigma)

    #sampler
    sampler = METROPOLIS(nwalkers=1000   , nstep=1000,
                         step_size = 0.5, nelec = wf.nelec, move = 'one',
                         ndim = wf.ndim, domain = {'min':-5,'max':5})

    # optimizer
    opt = optim.Adam(wf.parameters(),lr=0.005) # <- good for geo opt
    #opt = optim.Adam(wf.parameters(),lr=0.01) # <- good for coeff opt
    #opt = optim.SGD(wf.parameters(),lr=0.1)

    # network
    net = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)

    # single point
    #single_point(net,x=0.5,alpha=0.01)
    pos_curve(net)

    # geo opt
    #geo_opt(net,x=0.4)











