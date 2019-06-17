import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import torch.optim as optim

from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF_BASE
#from pyCHAMP.wavefunction.rbf import RBF1D as RBF
from pyCHAMP.wavefunction.rbf import RBF
from pyCHAMP.solver.deepqmc import DeepQMC
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS
from pyCHAMP.solver.mesh import adaptive_mesh_1d as mesh, fgradient, finverse
from pyCHAMP.solver.mesh import torchify

from pyCHAMP.solver.plot import plot_results_1d as plot_results
from pyCHAMP.solver.plot import plotter1d

import matplotlib.pyplot as plt

import numpy as np


class RBF_HO1D(NEURAL_WF_BASE):

    def __init__(self,nelec=1,ndim=1,ncenter=51):
        super(RBF_HO1D,self).__init__(nelec,ndim)

        # get the RBF centers 
        self.centers = torch.linspace(-5,5,ncenter).view(-1,1)

        # _func = torchify(self.nuclear_potential)
        # _fgrad = fgradient(_func)
        # _finv = finverse(_fgrad)

        # self.centers = torch.tensor(mesh(_finv,-5,5,ncenter,loss='default'))
        # self.centers = torch.sort(self.centers)[0]
        # self.centers = self.centers.view(-1,1)

        self.ncenter = len(self.centers)

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, self.ncenter,centers=self.centers, opt_centers=False)
        
        # define the fc layer
        self.fc = nn.Linear(self.ncenter, 1, bias=False)

        # initiaize the fc layer
        self.fc.weight.data.fill_(0.1)
        #self.fc.weight.data[0,1] = 1.
        #nn.init.uniform_(self.fc.weight,0,1)

    def forward(self,x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''

        batch_size = x.shape[0]
        #x = x.view(batch_size,-1,self.ndim)
        x = self.rbf(x)
        x = self.fc(x)
        return x.view(-1,1)

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi
        '''
        return (0.5*pos**2).flatten().view(-1,1)

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        return 0

def ho1d_sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    vn = np.exp(-0.5*pos**2)
    return vn/np.linalg.norm(vn)

# wavefunction
wf = RBF_HO1D(ndim=1,nelec=1,ncenter=5)

#sampler
sampler = METROPOLIS(nwalkers=250, nstep=1000, 
                     step_size = 3., nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-5,'max':5})

# optimizer
opt = optim.Adam(wf.parameters(),lr=0.005)

# network
net = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)
pos = None
obs_dict = None

# plt.ion()
# domain = {'xmin':-5.,'xmax':5.}
# plot1D = plotter1d(wf,domain,50,sol=ho1d_sol)



    # net.wf.fc.weight.requires_grad = True
    # net.wf.rbf.centers.requires_grad = False

    # pos,obs_dict = net.train(250,
    #          batchsize=250,
    #          pos = pos,
    #          obs_dict = obs_dict,
    #          resample=100,
    #          ntherm=-1,
    #          loss = 'variance',
    #          plot=plot1D,
    #          refine=25)

    # net.wf.fc.weight.requires_grad = False
    # net.wf.rbf.centers.requires_grad = True

    # pos,obs_dict = net.train(10,
    #          batchsize=250,
    #          pos = pos,
    #          obs_dict = obs_dict,
    #          resample=100,
    #          ntherm=-1,
    #          loss = 'variance',
    #          sol=ho1d_sol,
    #          fig=fig)

#plot_results(net,obs_dict,ho1d_sol,e0=0.5)






