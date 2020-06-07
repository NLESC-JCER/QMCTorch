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
from qmctorch.wavefunction.wf_FermiNet import FermiNet
from qmctorch.wavefunction.slater_pooling import SlaterPooling
from qmctorch.solver.solver_base import SolverBase
from qmctorch import log

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


class SolverFermiNet(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None,
                 scheduler=None, output=None, rank=0):

        SolverBase.__init__(self, wf, sampler,
                            optimizer, scheduler, output,  rank=0)

        self.mol = self.wf.mol
        self.save_model = "FermiNet_model.pth"
        self.configure()

    def configure(self):
        """Configure the solver for FermiNet optimization."""
        # opt all
        for param in self.wf.parameters():
            param.requires_grad = True

    def pretrain(self, nepoch, sampler=None, optimizer=None, load=None, with_tqdm=True):

        self.task = "Pretraining of FermiNet"
        loss_method = "MSE"

        # optimization method:
        self.opt = optimizer

        # keep track of loss:
        self.Loss_list = torch.zeros(nepoch)

        # optimization criterion:
        self.criterion = nn.MSELoss()

        if load is not None:
            self.load_checkpoint(load)

        # sampler for pretrianing
        if sampler is not None:
            self.sampler = sampler
        elif self.sampler is None:
            TypeError("No sampler was given.")

        # for the pre-trianing we will create a train orbital
        # using ground state config with a single determinant.
        self.hf_train = Orbital(
            self.mol, configs="ground_state", use_jastrow=False)

        # #initial position of the walkers

        pos = torch.cat((self.sampler(self.hf_train.pdf,
                                      with_tqdm=False),
                         self.sampler(self.hf_train.pdf,
                                      with_tqdm=False)), dim=0)

        # start pretraining
        min_loss = 1E5
        self.log_data_opt(nepoch, loss_method, pos.shape[0])
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
                                          with_tqdm=False)), dim=0)

            self.pretraining_epoch(pos)

            self.Loss_list[epoch] = self.loss.item()

            # keep track of how much time has elapsed
            elapsed = time.time() - start
            log.info('  elapsed time %.2f s' % elapsed)

            # save the model if necessary
            if self.loss < min_loss:
                min_loss = self.save_checkpoint(epoch,
                                                self.loss, self.save_model)

    def pretraining_epoch(self, pos):
        # optimization steps performed each epoch
        # get the predictions of the model and the training results of the orbitals to which we will train.
        MO_up, MO_down = self.hf_train._get_slater_matrices(pos)
        MO_up_fermi, MO_down_fermi = self.wf.compute_mo(pos)

        # detach training values:
        MO_up, MO_down = MO_up.repeat(1, self.wf.Kdet, 1, 1).detach(
        ), MO_down.repeat(1, self.wf.Kdet, 1, 1).detach()

        # --------------------------------------------------------------------- #
        # ----------------------[ Pretrain the FermiNet ]---------------------- #
        # --------------------------------------------------------------------- #

        self.opt.zero_grad()

        # calculate the loss and back propagate
        loss_up = self.criterion(MO_up_fermi, MO_up)
        loss_down = self.criterion(MO_down_fermi, MO_down)
        self.loss = (loss_up + loss_down) * 0.5
        log.options(style='percent').info(
            '  loss %f' % (self.loss))

        self.loss.backward()
        self.opt.step()

    def log_data_opt(self, nepoch, loss_method, nbatch):
        """Log data for the optimization."""
        log.info('  Task                :', self.task)
        log.info(
            '  Number Parameters   : {0}', self.wf.get_number_parameters())
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

    def plot_loss(self, path=None):

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
    
    @staticmethod
    def Display_orbital(wf, plane="z", plane_coord = 0.00, 
                    start = -5, end = 5, step = 0.1, 
                    wftype="Fermi", path=None, title=None):
        """"Function in attempt to visualise the orbital behaviour of the FermiNet.
            The function will display the first orbital of the first determinant.
            All electrons except one will be kept at a constant position.
            The output of the wave function will be determined over a 2D grid on a given plane.

            Args:
                wf (qmctorch.WaveFunction, optional): wave function. 
                plane (str, optional): The axis orthogonal to the grid plane.
                                        Default to "z": options ["x","y","z"]
                plane_coord (float, optional): Constant coordinate on the plane axis.
                                        Default to 0.00
                start (float, optional): Starting grid point. Default: -5
                end (float, optional): End grid point. Default: 5
                step (float, optional): step size of grid. Default: 0.1
                wftype (str, optional): wave function type: Default to "Fermi"
                                        Options: "Fermi", "Orbital"
                path (str, optional): path/filename to save the plot to   
                title (str, optional): title of the plot  
        """
        # keep all electrons except one at a constant position to:
        dim = ["x","y","z"]
        if plane not in dim: 
            ValueError("{} is not a valid plane. choose from {}.".format(plane,dim))
        plane_index = dim.index(plane)
        index =[0,1,2]
        index.pop(plane_index)

        grid = torch.arange(start ,end,
                            step, device="cpu")

        pos_1 = torch.zeros(len(grid), len(grid), 3)


        grid1, grid2 = torch.meshgrid(grid,grid)
        grid3 = plane_coord*torch.ones((grid1.shape[0],grid1.shape[1]))   
        grid12 = torch.cat((grid1.unsqueeze(2),grid2.unsqueeze(2)),dim=2)
        pos_1[:,:,index] = grid12
        pos_1[:,:,plane_index] = grid3
        pos_1 = pos_1.reshape(grid1.shape[0]**2,3)

        # all other electrons at constant position (1,1,1)
        pos = torch.ones((pos_1.shape[0],wf.mol.nelec,wf.ndim), device="cpu")  
        pos[:,0] = pos_1
        pos = pos.reshape((pos.shape[0],wf.mol.nelec*wf.ndim))

        if wftype == "Fermi":
            mo_up, mo_down = wf.compute_mo(pos)
        elif wftype == "Orbital":
            mo_up, mo_down = wf._get_slater_matrices(pos)
        else: 
            ValueError("The wftype {} is not a \
                        valid wf type.".format(wftype))
            
        mo_up = mo_up.detach().reshape((grid1.shape[0],
                            grid1.shape[0],mo_up.shape[1],
                            mo_up.shape[2], mo_up.shape[3]))


        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(grid1.numpy(), grid2.numpy(), mo_up[:,:,0,0,0].numpy())
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(dim[index[0]])
        ax.set_ylabel(dim[index[1]])
        ax.set_zlabel(r'$\phi({},{},{}={})$'.format(dim[index[0]], dim[index[1]], plane, plane_coord))

        if path is not None:
            fig.savefig(path)
        else :
            fig.show()


    def run(self):
        pass
