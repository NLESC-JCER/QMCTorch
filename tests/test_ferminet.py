import torch
from torch import nn 
from torch import optim
import sys 
sys.path.insert(0,'/home/breebaart/dev/QMCTorch/')
sys.path.insert(0,'/home/breebaart/dev/QMCTorch/tests/')

from qmctorch.wavefunction import Molecule
from qmctorch.wavefunction.intermediate_FermiNet import IntermediateLayers
from qmctorch.wavefunction.orbital_FermiNet import Orbital_FermiNet
from qmctorch.wavefunction.wf_FermiNet import FermiNet
from qmctorch.utils import set_torch_double_precision
from qmctorch.solver import SolverOrbital
from qmctorch.sampler import Metropolis
from qmctorch.utils import (plot_energy, plot_data)
# from qmctorch.wavefunction.FermiNet_v2 import FermiNet
import numpy as np 
import unittest

class TestFermiNet(unittest.TestCase):

    def setUp(self):
        # with a system in 3 dimensional space
        self.N_dim = 3
        set_torch_double_precision()
        # define the molecule
        # filename = "hdf5/H2_adf_dzp.hdf5"
        self.mol = mol = Molecule(atom='O	 0.000000 0.00000  0.00000; H 	 0.758602 0.58600  0.00000;H	-0.758602 0.58600  0.00000', 
                unit='bohr', calculator='pyscf', name='water')   
        # # filename = ["C1O2_adf_dzp.hdf5", "H1Li1_adf_dzp.hdf5", "H2_adf_dzp.hdf5"] 
        # self.mol = Molecule(load=filename)

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
        # print(Orbital)
        # # check the number of parameters and layers of the network:
        # for name, param in Orbital.named_parameters():
        #     print(name, param.size())

        # print(self.getNumParams(Orbital.parameters()))

        # # check the output of the network:
        # h_i, h_ij = FN.forward(self.r)
        # # print("h_i is given by: \n {}".format(h_i))
        # # and the output for the orbital
        # det = torch.zeros(self.batch, 3, 3)
        # orbital_j = Orbital.forward(h_i[:,0], self.r[:,0]).reshape(1,1,3)
        # print(orbital_j)
        # det[:,0,0] = orbital_j
    
    def test_FermiNet(self):
        
        WF = FermiNet(self.mol, self.hidden_nodes_e,
                    self.hidden_nodes_ee, self.L_layers, self.K_determinants)

        # check the number of parameters and layers of the intermediate layers:
        # for name, param in WF.named_parameters():
        #     print(name, param.size())

        # print(self.getNumParams(WF.parameters()))

        # # check the output of the network:
        # psi = WF.forward(self.r)
        # print(psi)

    def test_sampler(self):
        
        WF = FermiNet(self.mol, self.hidden_nodes_e,
                    self.hidden_nodes_ee, self.L_layers, self.K_determinants)
        
        # sampler
        sampler = Metropolis(nwalkers=200,
                            nstep=200, step_size=0.2,
                            ntherm=-1, ndecor=100,
                            nelec=WF.nelec, init=self.mol.domain('atomic'),
                            move={'type': 'all-elec', 'proba': 'normal'})
        
        print(WF.pdf(self.r))
        print(sampler(WF.pdf))

    # def test_optimization(self):
    #     WF = FermiNet(self.mol, 10, 4, 4, 
    #                   1, configs='ground_state', kinetic='jacobi', cuda=False)
        
    #     sampler = Metropolis(nwalkers=200,
    #                  nstep=200, step_size=0.2,
    #                  ntherm=-1, ndecor=100,
    #                  nelec=WF.nelec, init=self.mol.domain('atomic'),
    #                  move={'type': 'all-elec', 'proba': 'normal'})
        
    #     opt = optim.Adam(WF.parameters(), lr=1E-3)

        
    #     # scheduler
    #     scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.90)

    #     # QMC solver
    #     solver = SolverOrbital(wf=WF, sampler=sampler,
    #                         optimizer=opt, scheduler=None)

    #     # perform a single point calculation
    #     obs = solver.single_point()

    #     # optimize the wave function
    #     solver.configure(task='wf_opt')
    #     solver.track_observable(['local_energy'])

    #     solver.configure_resampling(mode='update',
    #                                 resample_every=1,
    #                                 nstep_update=50)
    #     solver.ortho_mo = False
    #     obs = solver.run(250, batchsize=None,
    #                     loss='energy',
    #                     grad='auto',
    #                     clip_loss=False)

    #     plot_energy(obs.local_energy, e0=-1.1645, show_variance=True)
    #     plot_data(solver.observable, obsname='jastrow.weight')

 

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
    # t.test_intermediate()
    # t.test_orbital()
    # t.test_FermiNet()
    # t.test_sampler()
    # t.test_optimization()



