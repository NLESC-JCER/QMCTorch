# this file will contain the function for pretraining_steps of the FermiNet.
# the FermiNet is pretrained to pyscf sto-3g hf orbitals.
# this pretraining reduces the variance of the calculations when optimizing the FermiNet
# and allows to skip the more non-physical regions in the optimization.

# Fermi Orbital with own Parameter matrices
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from qmctorch.wavefunction import Orbital, Molecule
from qmctorch.solver import SolverOrbital
from qmctorch.wavefunction.orbital_projector import OrbitalProjector
from qmctorch.sampler import Metropolis
from qmctorch.utils import set_torch_double_precision
from qmctorch.utils import (plot_energy, plot_data)
from qmctorch.utils import (
    DataSet, Loss, OrthoReg, dump_to_hdf5, add_group_attr)
from qmctorch.utils.plot_mo import Display_orbital
from qmctorch.wavefunction import WaveFunction
from qmctorch.wavefunction.wf_FermiNet import FermiNet
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
        self.task = "wf_opt"
        self.configure()
        self.loss = None

    def configure(self):
        """Configure the solver for FermiNet optimization."""
        # opt all
        for param in self.wf.parameters():
            param.requires_grad = True

    def pretrain(self, nepoch, optimizer=None, load=None, 
                 with_tqdm=True, display_every=None,
                 path_figure=None):

        self.task = "Pretraining of FermiNet"
        loss_method = "MSE"

        # optimization method:
        self.opt = optimizer

        # keep track of loss:
        self.Loss_list = torch.zeros(nepoch)

        # optimization criterion:
        self.criterion = nn.MSELoss()

        load_epoch = 0
        if load is not None:
            load_epoch, _ = self.load_checkpoint(load)
            log.info(' Using loaded Network')

        # for the pre-trianing we will create a train orbital
        # using ground state config with a single determinant.
        self.hf_train = Orbital(
            self.mol, configs="ground_state", use_jastrow=False)
        
        # initial position of the walkers
        pos = self.sampler(self.hf_train.pdf, with_tqdm=False)

        # change the number of steps/walker size
        _nstep_save = self.sampler.nstep
        _ntherm_save = self.sampler.ntherm
        _nwalker_save = self.sampler.walkers.nwalkers
        if self.resampling_options.mode == 'update':
            self.sampler.ntherm = -1
            self.sampler.nstep = self.resampling_options.nstep_update
            self.sampler.walkers.nwalkers = int(pos.shape[0]/2)
            self.sampler.nwalkers = int(pos.shape[0]/2)
        

        # start pretraining
        min_loss = 1E5
        self.log_data_opt(nepoch, loss_method, pos.shape[0])
        start = time.time()
        for epoch in range(load_epoch, nepoch+load_epoch):
            log.info(' ')
            log.info('  epoch %d' % epoch)
            # sample from both the hf and FermiNet
            # take 10 Metropolis-Hastings steps
            if epoch % self.resampling_options.nstep_update == 0:
                pos = torch.cat((self.sampler(self.hf_train.pdf,
                                            pos[:self.sampler.nwalkers],
                                            with_tqdm=False),
                                self.sampler(self.wf.pdf,
                                            pos[self.sampler.nwalkers:],
                                            with_tqdm=False)), dim=0)

            self.pretraining_epoch(pos)

            self.Loss_list[epoch-load_epoch] = self.loss.item()

            # keep track of how much time has elapsed
            elapsed = time.time() - start
            log.info('  elapsed time %.2f s' % elapsed)

            if display_every is not None:
                if epoch % display_every == 0:
                    Display_orbital(self.wf.compute_mo, self.wf, 
                                path=path_figure+str(epoch), 
                                title="Pretraining FermiNet epoch: {}".format(epoch))

            # save the model if necessary
            if self.loss < min_loss:
                min_loss = self.save_checkpoint(epoch,
                                                self.loss, self.save_model)
        
        # restore the sampler number of step
        self.sampler.nstep = _nstep_save
        self.sampler.ntherm = _ntherm_save
        self.sampler.walkers.nwalkers = _nwalker_save
        self.sampler.nwalkers = _nwalker_save

        log.info(' ')
        log.info(' Finished pretraining FermiNet')

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
        grad = "auto"
        log.info('  Task                :', self.task)
        log.info(
            '  Number Parameters   : {0}', self.wf.get_number_parameters())
        log.info('  Number of epoch     : {0}', nepoch)
        log.info('  Batch size          : {0}', nbatch)
        log.info('  Loss function       : {0}', loss_method)
        if hasattr(self.loss, 'clip'):
            log.info('  Clip Loss           : {0}', self.loss.clip)
        log.info('  Gradients           : {0}', grad)
        log.info('  Resampling mode     : {0}', self.resampling_options.mode)
        log.info(
            '  Resampling every    : {0}', self.resampling_options.resample_every)
        log.info(
            '  Resampling steps    : {0}', self.resampling_options.nstep_update)
        log.info('')

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
 
   
    def run(self, nepoch, batchsize=None, loss='energy',
            clip_loss=False, grad='auto', hdf5_group=None):
        """Run the optimization

        Args:
            nepoch (int): Number of optimziation step
            batchsize (int, optional): Number of sample in a mini batch.
                                       If None, all samples are used.
                                       Defaults to None.
            loss (str, optional): merhod to compute the loss: variance or energy.
                                  Defaults to 'energy'.
            clip_loss (bool, optional): Clip the loss values at +/- 5std.
                                        Defaults to False..
            grad (str, optional): method to compute the gradients: 'auto' or 'manual'.
                                  Defaults to 'auto'.
            hdf5_group (str, optional): name of the hdf5 group where to store the data.
                                        Defaults to wf.task.
        """
        self.task ="wf_opt"

        log.info('')
        log.info('  {0} optimization', loss)

        # observable
        if not hasattr(self, 'observable'):
            self.track_observable(['local_energy'])

        self.evaluate_gradient = {
            'auto': self.evaluate_grad_auto,
            'manual': self.evaluate_grad_manual}[grad]

        if 'lpos_needed' not in self.opt.__dict__.keys():
            self.opt.lpos_needed = False

        # get the loss
        self.loss = Loss(self.wf, method=loss, clip=clip_loss)
        self.loss.use_weight = (
            self.resampling_options.resample_every > 1)

        # log data
        self.log_data_opt(nepoch, loss, batchsize)

        # sample the wave function
        pos = self.sampler(self.wf.pdf)

        # handle the batch size
        if batchsize is None:
            batchsize = len(pos)

        # get the initial observable
        self.store_observable(pos)

        # change the number of steps/walker size
        _nstep_save = self.sampler.nstep
        _ntherm_save = self.sampler.ntherm
        _nwalker_save = self.sampler.walkers.nwalkers
        if self.resampling_options.mode == 'update':
            self.sampler.ntherm = -1
            self.sampler.nstep = self.resampling_options.nstep_update
            self.sampler.walkers.nwalkers = pos.shape[0]
            self.sampler.nwalkers = pos.shape[0]

        # create the data loader
        self.dataset = DataSet(pos)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batchsize)

        cumulative_loss = []
        min_loss = 1E3

        # loop over the epoch
        start = time.time()
        for n in range(nepoch):

            log.info('  epoch %d' % n)

            cumulative_loss = 0

            # loop over the batches
            for ibatch, data in enumerate(self.dataloader):

                # port data to device
                lpos = data.to(self.device)

                # get the gradient
                loss, eloc = self.evaluate_gradient(lpos)
                cumulative_loss += loss

                # optimize the parameters
                self.optimization_step(lpos)

                # observable
                self.store_observable(
                    pos, local_energy=eloc, ibatch=ibatch)

            # save the model if necessary
            if cumulative_loss < min_loss:
                min_loss = self.save_checkpoint(
                    n, cumulative_loss, self.save_model)

            self.print_observable(cumulative_loss)

            # keep track of how much time has elapsed
            elapsed = time.time() - start
            log.info('  elapsed time %.2f s' % elapsed)
            log.info('')

            # resample the data
            pos = self.resample(n, pos)

            if self.scheduler is not None:
                self.scheduler.step()

 


        # restore the sampler number of step
        self.sampler.nstep = _nstep_save
        self.sampler.ntherm = _ntherm_save
        self.sampler.walkers.nwalkers = _nwalker_save
        self.sampler.nwalkers = _nwalker_save

        # dump
        if hdf5_group is None:
            hdf5_group = self.task
        dump_to_hdf5(self.observable, self.hdf5file, hdf5_group)
        add_group_attr(self.hdf5file, hdf5_group, {'type': 'opt'})

        return self.observable
    
    def evaluate_grad_auto(self, lpos):
        """Evaluate the gradient using automatic differentiation

        Args:
            lpos (torch.tensor): sampling points

        Returns:
            tuple: loss values and local energies
        """

        # compute the loss
        loss, eloc = self.loss(lpos)

        # compute local gradients
        self.opt.zero_grad()
        loss.backward()

        return loss, eloc
    
    
    def evaluate_grad_manual(self, lpos):
        """Evaluate the gradient using low variance express

        Args:
            lpos ([type]): [description]

        Args:
            lpos (torch.tensor): sampling points

        Returns:
            tuple: loss values and local energies
        """

        if self.loss.method in ['energy', 'weighted-energy']:

            ''' Get the gradient of the total energy
            dE/dk = 2 < (dpsi/dk)/psi (E_L - <E_L >) >
            '''

            # compute local energy and wf values
            _, eloc = self.loss(lpos, no_grad=False)
            eloc = torch.tensor(eloc.clone().detach(), requires_grad=False)
            psi = self.wf(lpos)
            norm = 1. / len(psi)

            # evaluate the prefactor of the grads
            weight = eloc.clone()
            weight -= torch.mean(eloc)
            weight /= psi.clone().detach()
            weight *= 2.
            weight *= norm

            # compute the gradients
            self.opt.zero_grad()
            psi.backward(weight)

            return torch.mean(eloc), eloc

        else:
            raise ValueError(
                'Manual gradient only for energy minimization')