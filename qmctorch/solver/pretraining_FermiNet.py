# this file will contain the function for pretraining_steps of the FermiNet.
# the FermiNet is pretrained to pyscf sto-3g hf orbitals.
# this pretraining reduces the variance of the calculations when optimizing the FermiNet
# and allows to skip the more non-physical regions in the optimization.

# Fermi Orbital with own Parameter matrices 
import torch 
from torch import nn 
import sys
sys.path.insert(0,'/home/breebaart/dev/QMCTorch/')

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.solver import SolverOrbital
from qmctorch.sampler import Metropolis
from qmctorch.utils import set_torch_double_precision
from qmctorch.utils import (plot_energy, plot_data)
from qmctorch.wavefunction import WaveFunction
from qmctorch.wavefunction.FermiNet_v2 import FermiNet

import numpy as np 

class solver_FermiNet(object):
    
    def __init__(self, wf=None, sampler=None,
                 optimizer=None, scheduler=None,
                 output=None, rank=0):
        
        self.wf = wf
        self.sampler = sampler
        self.opt = optimizer
        self.scheduler = scheduler
        self.cuda = False
        self.device = torch.device('cpu')


    
    def pretraining(self, epochs):


        # this funciton will run the pretrianing 


    def pretraining_epoch(self):



if __name__ == "__main__":
    



