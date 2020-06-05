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
from qmctorch.solver.solver_base import SolverBase
from qmctorch import log

import numpy as np 
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

class SolverFermiNet(SolverBase):
    
    def __init__(self,wf=None, sampler=None, optimizer=None, 
                    scheduler=None, output=None, rank=0):

        SolverBase.__init__(self, wf, sampler,
                            optimizer, scheduler, output, task ="fermi_opt", rank=0)
        
        self.mol = self.wf.mol
        self.save_model ="FermiNet_model.pth"
    
    def pretrain(self, nepoch, sampler=None, optimizer=None, load=None, with_tqdm = True):
        
        self.task ="Pretraining of FermiNet"
        loss_method= "MSE"

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
        
        pos = torch.cat((self.sampler(self.hf_train.pdf,
                with_tqdm=False),
                self.sampler(self.hf_train.pdf,
                with_tqdm=False)),dim=0)

        # start pretraining    
        min_loss = 1E5
        self.log_data_opt(nepoch,loss_method,pos.shape[0])
        start = time.time()
        for epoch in range(nepoch):
            # sample from both the hf and FermiNet switching every epcoch
            # take 10 Metropolis-Hastings steps
            log.info(' ') 
            log.info('  epoch %d' % epoch)

            # if epoch % 2:
            #      pos = self.sampler(self.wf.pdf,pos,with_tqdm=False)
            # else:
            #     pos = self.sampler(self.hf_train.pdf,pos,with_tqdm=False)

            pos = torch.cat((self.sampler(self.hf_train.pdf,
                pos[:self.sampler.nwalkers],
                with_tqdm=False),
                self.sampler(self.wf.pdf,
                pos[self.sampler.nwalkers:],
                with_tqdm=False)),dim=0)
          
            self.pretraining_epoch(pos) 

            self.Loss_list[epoch] = self.loss.item()
            
            # keep track of how much time has elapsed 
            elapsed = time.time() - start
            log.info('  elapsed time %.2f s' % elapsed)


            # save the model if necessary
            if self.loss < min_loss:
                min_loss = self.save_checkpoint(epoch,
                            self.loss, self.save_model) 


                    

    def pretraining_epoch(self,pos):
        # optimization steps performed each epoch
        # get the predictions of the model and the training results of the orbitals to which we will train.
        MO_up, MO_down = self.hf_train._get_slater_matrices(pos)
        MO_up_fermi, MO_down_fermi = self.wf.compute_mo(pos)

        # detach training values:
        MO_up, MO_down = MO_up.repeat(1, self.wf.Kdet, 1, 1).detach(), MO_down.repeat(1, self.wf.Kdet, 1, 1).detach()

        # --------------------------------------------------------------------- #
        # ----------------------[ Pretrain the FermiNet ]---------------------- #
        # --------------------------------------------------------------------- #
        
        self.opt.zero_grad()

        #calculate the loss and back propagate 
        loss_up = self.criterion(MO_up_fermi,MO_up)
        loss_down = self.criterion(MO_down_fermi, MO_down)
        self.loss = (loss_up + loss_down) * 0.5
        log.options(style='percent').info(
                    '  loss %f' % (self.loss))

        self.loss.backward()
        self.opt.step()  


    def log_data_opt(self, nepoch, loss_method, nbatch):
        """Log data for the optimization."""
        log.info('  Task                :', self.task)
        log.info('  Number Parameters   : {0}', self.wf.get_number_parameters())
        log.info('  Number of epoch     : {0}', nepoch)
        log.info('  Batch size          : {0}', nbatch)
        log.info('  Loss function       : {0}', loss_method)
        # log.info('  Clip Loss           : {0}', self.loss.clip)
        # log.info('  Gradients           : {0}', grad)
        # log.info('  Resampling mode     : {0}', self.resampling_options.mode)
        # log.info(
        #     '  Resampling every    : {0}', self.resampling_options.resample_every)
        # log.info(
        #     '  Resampling steps    : {0}', self.resampling_options.nstep_update)
        # log.info('')
       
    def save_loss_list(self, filename):
        torch.save(self.Loss_list, filename)

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
