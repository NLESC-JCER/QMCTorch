import torch
from torch import nn 
from torch import optim

from qmctorch.wavefunction import Molecule
from qmctorch.utils import set_torch_double_precision
from qmctorch.solver import SolverOrbital
from qmctorch.sampler import Metropolis
from qmctorch.utils import (plot_energy, plot_data)
from qmctorch.wavefunction.wf_FermiNet import FermiNet

import numpy as np 
import unittest

class TestFermiNet(unittest.TestCase):

    def setUp(self):
        # with a system in 3 dimensional space
        self.N_dim = 3
        set_torch_double_precision()
        # define the molecule
        self.mol = mol = Molecule(atom='O	 0.000000 0.00000  0.00000; H 	 0.758602 0.58600  0.00000;H	-0.758602 0.58600  0.00000', 
                unit='bohr', calculator='pyscf', name='water')   

        # network hyperparameters: 
        self.hidden_nodes_e = 256
        self.hidden_nodes_ee = 32
        self.K_determinants = 1
        self.L_layers = 4

        # set a initial seed for to make the example reproducable
        torch.random.manual_seed(321)
        self.batch = 3
        # initiaite a random configuration of particle positions
        self.r = torch.randn(self.batch,self.mol.nelec, self.N_dim, device="cpu")
        # self.r = torch.ones((self.batch,self.mol.nelec, self.N_dim), device="cpu")
    
    def test_FermiNet(self):
        
        WF = FermiNet(self.mol, self.hidden_nodes_e,
                    self.hidden_nodes_ee, self.L_layers, self.K_determinants)

        # check the number of parameters and layers of the intermediate layers:
        # for name, param in WF.named_parameters():
        #     print(name, param.size())

        # print(self.getNumParams(WF.parameters()))

        # # check the output of the network:
        psi = WF.forward(self.r)

    def test_sampler(self):
        
        WF = FermiNet(self.mol, self.hidden_nodes_e,
                    self.hidden_nodes_ee, self.L_layers, self.K_determinants)
        
        # sampler
        sampler = Metropolis(nwalkers=1,
                            nstep=20000, step_size=0.2,
                            ntherm=-1, ndecor=100,
                            nelec=WF.nelec, init=self.mol.domain('atomic'),
                            move={'type': 'all-elec', 'proba': 'normal'})
        
        WF.pdf(self.r)
        sampler(WF.pdf)

    @staticmethod
    def getNumParams(params):
        '''function to get the variable count
        from https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/12'''
        numParams, numTrainable = 0, 0
        for param in params:
            npParamCount = np.prod(param.data.shape)
            numParams += npParamCount
            if param.requires_grad:
                numTrainable += npParamCount
        return numParams, numTrainable      

if __name__ == "__main__":
    unittest.main()
    # t = TestFermiNet()
    # t.setUp()
    # t.test_FermiNet()
    # t.test_sampler()
    # t.test_optimization()



