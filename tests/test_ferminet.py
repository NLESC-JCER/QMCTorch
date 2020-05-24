import torch 
from torch import nn 

from qmctorch.wavefunction import Molecule
from qmctorch.wavefunction.intermediate_FermiNet import IntermediateLayers

import numpy as np 
import unittest

class TestFermiNet(unittest.TestCase):

    def __init__(self):
                # for this initial example I will start with a smaller 4 particle system with 2 spin up and 2 spin down particles.
        self.N_spin = [2, 2]
        self.N_electrons = np.sum(self.N_spin)
        # with a system in 3 dimensional space
        self.N_dim = 3

        self.mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
               calculator='pyscf',
               basis='sto-3g',
               unit='bohr')

        # network hyperparameters: 
        self.hidden_nodes_e = 256
        self.hidden_nodes_ee = 32
        self.K_determinants = 1
        self.L_layers = 4

        # set a initial seed for to make the example reproducable
        torch.random.manual_seed(321)

        # initiaite a random configuration of particle positions
        self.r = torch.randn(self.N_electrons, self.N_dim, device="cpu")


    def test_intermediate(self):
                        
        # test the intermediate layers .
        FN = IntermediateLayers(self.mol, self.N_spin, self.hidden_nodes_e, self.hidden_nodes_ee, self.L_layers)
        print(FN)
        # check the number of parameters and layers of the intermediate layers:

        for name, param in FN.named_parameters():
            print(name, param.size())

        print(self.getNumParams(FN.parameters()))

        print(FN.forward(self.r))
    
    def test_orbital(self):

        # test the orbital
        Orbital = Orbital_k_i()
        print(Orbital)
        print(Orbital.forward(self.r))


    @staticmethod
    def getNumParams(params):
        '''function to get the variable count'''
        numParams, numTrainable = 0, 0
        for param in params:
            npParamCount = np.prod(param.data.shape)
            numParams += npParamCount
            if param.requires_grad:
                numTrainable += npParamCount
        return numParams, numTrainable

#from https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/12


        



if __name__ == "__main__":
    # unittest.main()
    t = TestFermiNet()
    t.test_intermediate()
    t.test_orbital()



