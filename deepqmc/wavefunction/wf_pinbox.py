import numpy as np
import torch
from torch import nn

from deepqmc.wavefunction.wf_base import WF_BASE
from deepqmc.wavefunction.rbf import RBF
from deepqmc.wavefunction.mesh_utils import regular_mesh_2d, regular_mesh_3d

class PinBox(WF_BASE):

    def __init__(self,fpot,domain,ncenter,nelec=1):
        super(PinBox,self).__init__(nelec,len(ncenter))

        # get the RBF centers 
        ndim = len(ncenter)
        if ndim == 1:
            self.centers = torch.linspace(domain['xmin'],domain['xmax'],ncenter[0]).view(-1,1)
        elif ndim == 2:
            points = regular_mesh_2d(xmin=domain['xmin'],xmax=domain['xmax'],nx=ncenter[0],
                           ymin=domain['ymin'],ymax=domain['ymax'],ny=ncenter[1]) 
            self.centers = torch.tensor(points)
        self.ncenter = len(self.centers)

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, self.ncenter,
                      centers=self.centers, opt_centers=False,
                      sigma = 1.)
        
        # define the fc layer
        self.fc = nn.Linear(self.ncenter, 1, bias=False)

        # initiaize the fc layer
        nn.init.uniform_(self.fc.weight,0,1)

        # book the potential function
        self.user_potential = fpot

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
        return self.user_potential(pos).flatten().view(-1,1)

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        return 0

    def nuclear_repulsion(self):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        return 0







