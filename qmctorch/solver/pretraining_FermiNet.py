# this file will contain the function for pretraining_steps of the FermiNet.
# the FermiNet is pretrained to pyscf sto-3g hf orbitals.
# this pretraining reduces the variance of the calculations when optimizing the FermiNet
# and allows to skip the more non-physical regions in the optimization.

# Fermi Orbital with own Parameter matrices 
import torch 
from torch import nn 
from torch import optim 

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.solver import SolverOrbital
from qmctorch.wavefunction.orbital_projector import OrbitalProjector
from qmctorch.sampler import Metropolis
from qmctorch.utils import set_torch_double_precision
from qmctorch.utils import (plot_energy, plot_data)
from qmctorch.wavefunction import WaveFunction
from qmctorch.wavefunction.FermiNet_v2 import FermiNet
from qmctorch.wavefunction.slater_pooling import SlaterPooling

import numpy as np 
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

class SolverFermiNet():
    
    def __init__(self,wf=None, sampler=None, scheduler=None):
        
        self.wf = wf
        self.mol = self.wf.mol
        self.sampler = sampler
        # ADAM 
        self.scheduler = scheduler
        self.cuda = False
        self.device = torch.device('cpu')
    
    def pretrain(self, nepoch, sampler=None, optimizer=None, load=None, with_tqdm = True):
        
        # optimization method:
        self.opt = optimizer

        # keep track of loss:
        self.Loss_list = torch.zeros(nepoch)

        # optimization criterion:
        self.criterion = nn.MSELoss()
        
        # sampler for pretrianing
        if sampler is not None: 
                self.sampler =sampler
        elif self.sampler is None:
            TypeError("No sampler was given.")
        
        # for the pre-trianing we will create a train orbital 
        # using ground state config with a single determinant.
        self.hf_train = Orbital(self.mol, configs = "ground_state", use_jastrow=False)    
        
        # #initial position of the walkers
        # pos = self.sampler(self.wf.pdf) 
        pos = torch.randn(self.sampler.nwalkers,self.mol.nelec*3)
        
        # start pretraining    
        start = time.time()
        # this function will run the pretrianing 
        epochs = tqdm(range(nepoch),
                desc='INFO:QMCTorch|  Pretraining',
                disable=not with_tqdm)
        for epoch in epochs:
            # sample from both the hf and FermiNet switching every epcoch
            # take 10 Metropolis-Hastings steps
            if epoch % 2:
                # print("FERMI NET SAMPLING")
                pos = self.sampler(self.wf.pdf,pos,with_tqdm=False)
            else:
                pos = self.sampler(self.hf_train.pdf,pos,with_tqdm=False) 

            self.pretraining_epoch(pos)  
            self.Loss_list[epoch] = self.Loss.item()
            
            # keep track of how much time has elapsed 
            elapsed = time.time() - start

            # Show progress every 20 epoch 
            if not epoch % 20:
                 print(f'epoch: {epoch}, time: {elapsed:.3f}s, loss: {self.Loss.item():.3f}')
        
        # final result:
        print(f'epoch: {epoch}, time: {elapsed:.3f}s, loss: {self.Loss.item():.3f}')               

    def pretraining_epoch(self,pos):
        # optimization steps performed each epoch
        # get the predictions of the model and the training results of the orbitals to which we will train.
        MO_up, MO_down = self.hf_train._get_slater_matrices(pos)
        MO_up_fermi, MO_down_fermi = self.wf.forward_det(pos)

        # detach training values:
        MO_up, MO_down = MO_up.repeat(1, self.wf.Kdet, 1, 1).detach(), MO_down.repeat(1, self.wf.Kdet, 1, 1).detach()

        # print("fermi up:",MO_up_fermi.shape)
        # print("fermi down:",MO_down_fermi.shape)
        # print("MO up:",MO_up.shape)
        # print("MO down:",MO_down.shape)
        
        # --------------------------------------------------------------------- #
        # ----------------------[ Pretrain the FermiNet ]---------------------- #
        # --------------------------------------------------------------------- #
        
        self.opt.zero_grad()

        #calculate the loss and back propagate 
        Loss_up = self.criterion(MO_up_fermi,MO_up)
        Loss_down = self.criterion(MO_down_fermi, MO_down)
        self.Loss = (Loss_up + Loss_down) * 0.5


        self.Loss.backward()
        self.opt.step()  

    def plot_loss(self,path=None):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        n = len(self.Loss_list)
        epoch = np.arange(n)

        # plot
        ax.plot(epoch, self.Loss_list, color='#144477')
        ax.grid()
        ax.set_xlabel('Number of epoch')
        ax.set_ylabel('Loss', color='black')

        if path is not None:
            plt.savefig(path)
        else:
            plt.show()
        

    def run(self):
        pass

if __name__ == "__main__":
    # bond distance : 0.74 A -> 1.38 a
    # optimal H positions +0.69 and -0.69
    # ground state energy : -31.688 eV -> -1.16 hartree
    # bond dissociation energy 4.478 eV -> 0.16 hartree
    set_torch_double_precision()

    ndim =3
    nbatch =20

    # define the molecule
    mol = Molecule(atom='O 0.000 0.000 0.000; H 0.00000 0.0000 0.69; H 0.00000 0.0000 -0.69',
                calculator='pyscf',
                basis='sto-3g',
                unit='bohr')
    
    # create the network
    wf = FermiNet(mol,hidden_nodes_e=254,hidden_nodes_ee=32,L_layers=4,Kdet=4)

    # choose a sampler:
    sampler = Metropolis(nwalkers=nbatch,
                        nstep=10, step_size=0.2,
                        ntherm=-1, ndecor=100,
                        nelec=wf.nelec, init=mol.domain('atomic'),
                        move={'type': 'all-elec', 'proba': 'normal'})
    
    # choose a optimizer
    opt = optim.Adam(wf.parameters(), lr=1E-3)

    # set a initial seed to make the example reproducable
    torch.random.manual_seed(321)

    # initialize solver
    solverFermi = SolverFermiNet(wf,sampler=sampler)

    # pretrain the FermiNet to hf orbitals
    solverFermi.pretrain(100,optimizer=opt)
    solverFermi.plot_loss(path="/home/breebaart/dev/QMCTorch/TrainingExaminig/Figures/pretraining_loss")
    