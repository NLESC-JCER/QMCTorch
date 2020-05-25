import torch 
from torch import nn 

from qmctorch.wavefunction import Molecule
from qmctorch.wavefunction.intermediate_FermiNet import IntermediateLayers
from qmctorch.wavefunction.orbital_ferminet import Orbital_FermiNet

import numpy as np 
import unittest

class TestFermiNet(unittest.TestCase):

    def __init__(self):
        # with a system in 3 dimensional space
        self.N_dim = 3


        # define the molecule
        # self.mol = mol = Molecule(atom='O	 0.000000 0.00000  0.00000; H 	 0.758602 0.58600  0.00000;H	-0.758602 0.58600  0.00000', 
        #         unit='bohr', calculator='pyscf', name='water')   
        filename = ["C1O2_adf_dzp.hdf5", "H1Li1_adf_dzp.hdf5", "H2_adf_dzp.hdf5"] 
        self.mol = Molecule(load=filename[2])

        # network hyperparameters: 
        self.hidden_nodes_e = 256
        self.hidden_nodes_ee = 32
        self.K_determinants = 1
        self.L_layers = 4

        # set a initial seed for to make the example reproducable
        torch.random.manual_seed(321)

        # initiaite a random configuration of particle positions
        self.r = torch.randn(self.mol.nelec, self.N_dim, device="cpu")



    def test_intermediate(self):
                        
        # test the intermediate layers .
        FN = IntermediateLayers(self.mol, self.hidden_nodes_e, self.hidden_nodes_ee, self.L_layers)
        # print(FN)
        # # check the number of parameters and layers of the intermediate layers:

        # for name, param in FN.named_parameters():
        #     print(name, param.size())

        # print(self.getNumParams(FN.parameters()))

        print(FN.forward(self.r))
    
    def test_orbital(self):

        # test the orbital
        
        FN = IntermediateLayers(self.mol, self.hidden_nodes_e, 
                self.hidden_nodes_ee, self.L_layers)

        Orbital = Orbital_FermiNet(self.mol,self.hidden_nodes_e)
        print(Orbital)
        # check the number of parameters and layers of the network:
        for name, param in Orbital.named_parameters():
            print(name, param.size())

        print(self.getNumParams(Orbital.parameters()))

        # check the output of the network:
        h_i, h_ij = FN.forward(self.r)
        print("h_i is given by: \n {}".format(h_i))
        # and the output for the orbital
        print(Orbital.forward(h_i[0], r[0]))
  

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
    # unittest.main()
    t = TestFermiNet()
    t.test_intermediate()
    # t.test_orbital()



