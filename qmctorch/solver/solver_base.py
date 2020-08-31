import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace
from copy import deepcopy
from tqdm import tqdm
from time import time
import numpy as np
from ..sampler import Resampler
from ..utils import dump_to_hdf5, add_group_attr
from ..utils import DataSet, Loss
from .. import log
from .observable import Observable


class SolverBase(object):

    def __init__(self, wf=None, sampler=None,
                 optimizer=None, scheduler=None,
                 output=None, rank=0):
        """Base Class for QMC solver 

        Args:
            wf (qmctorch.WaveFunction, optional): wave function. Defaults to None.
            sampler (qmctorch.sampler, optional): Sampler. Defaults to None.
            optimizer (torch.optim, optional): optimizer. Defaults to None.
            scheduler (torch.optim, optional): scheduler. Defaults to None.
            output (str, optional): hdf5 filename. Defaults to None.
            rank (int, optional): rank of he process. Defaults to 0.
        """

        self.wf = wf
        self.sampler = sampler
        self.opt = optimizer
        self.scheduler = scheduler

        # cuda capabilities
        self.cuda = False
        self.device = torch.device('cpu')

        # handles GPU availability
        if self.wf.cuda:
            self.device = torch.device('cuda')
            self.sampler.cuda = True
            self.sampler.walkers.cuda = True
        else:
            self.device = torch.device('cpu')

        # name of the hdf5 file
        self.hdf5file = output
        if output is None:
            basename = self.wf.mol.hdf5file.split('.')[0]
            self.hdf5file = basename + '_QMCTorch.hdf5'

        # observable we want to track
        self.observable = Observable(
            ['local_energy', 'parameters'], self.wf)

        # resampler
        self.resampler = Resampler(sampler)

        # additional penalties on the loss
        self.loss = Loss(self.wf)

        # gradients
        self.grad_method = 'auto'
        self.evaluate_gradients = self.evaluate_grad_auto

        if rank == 0:
            dump_to_hdf5(self, self.hdf5file)

        self.log_data()

    def configure_observable(self, obsname):
        """Configure the observable we need

        Args:
            obsname (list ,str): names of the observables
        """
        self.observable = Observable(obsname, self.wf)

    def configure_resampling(self, mode, resample_every, nstep_update):
        """Configure the resampling

        Args:
            mode (str, optional): method to resample : 'full', 'update', 'never' 
                                  Defaults to 'update'.
            resample_every (int, optional): Number of optimization steps between resampling
                                 Defaults to 1.
            nstep_update (int, optional): Number of MC steps in update mode. 
                                          Defaults to 25.
        """
        self.resampler = Resampler(mode, resample_every, nstep_update)

    def run(self, nepoch, batchsize=None,
            hdf5_group=None, chkpt_every=None):
        """Run a wave function optimization

        Args:
            nepoch (int): Number of optimziation step
            batchsize (int, optional): Number of sample in a mini batch.
                                       If None, all samples are used.
                                       Defaults to Never.
            hdf5_group (str, optional): name of the hdf5 group where to store the data.
                                        Defaults to wf.task.
            chkpt_every (int, optional): save a checkpoint every every iteration.
                                         Defaults to half the number of epoch
        """

        # prepare the optimization
        self.prepare_optimization(batchsize, chkpt_every)
        self.log_data_opt(nepoch)

        # run the epochs
        self.run_epochs(nepoch)

        # dump
        self.observable.save(hdf5_group or 'wf_opt', self.hdf5file)

        return self.observable

    def prepare_optimization(self, batchsize, chkpt_every):
        """Prepare the optimization process

        Args:
            batchsize (int or None): batchsize
            chkpt_every (int or none): save a chkpt file every 
        """

        # sample the wave function
        pos = self.sampler(self.wf.pdf)

        # handle the batch size
        if batchsize is None:
            batchsize = self.sampler.nsample

        # get the initial observable
        self.observable.store(self.wf, pos)

        # create the data loader
        self.dataset = DataSet(pos)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batchsize)

        # chkpt
        self.chkpt_every = chkpt_every

    def run_epochs(self, nepoch):
        """Run a certain number of epochs

        Args:
            nepoch (int): number of epoch to run
        """

        # loop over the epoch
        for n in range(nepoch):

            tstart = time()
            log.info('')
            log.info('  epoch %d' % n)

            cumulative_loss = 0

            # loop over the batches
            for ibatch, data in enumerate(self.dataloader):

                # port data to device
                t0_opt = time()
                lpos = data.to(self.device)

                # get the gradient
                loss, eloc = self.evaluate_gradient(lpos)
                cumulative_loss += loss

                # optimize the parameters
                self.opt.step()

                # observable
                self.observable.store(self.wf,
                                      lpos,
                                      local_energy=eloc,
                                      ibatch=ibatch)

                log.info('  optmization step in %1.2f sec.' %
                         ((time()-t0_opt)))

            # save the model if necessary
            if n == 0 or cumulative_loss < min_loss:
                min_loss = cumulative_loss
                self.observable.store_model = dict(
                    self.wf.state_dict())

            # save checkpoint file
            if self.chkpt_every is not None:
                if (n > 0) and (n % self.chkpt_every == 0):
                    self.save_checkpoint(n, cumulative_loss)

            self.observable.print(cumulative_loss)

            # resample the data
            t0_resample = time()
            self.dataset.data = self.resampler(
                self.wf.pdf, n, self.dataset.data)
            self.loss.weight['psi0'] = None

            log.info('  resampling done in %1.2f sec.' %
                     (time()-t0_resample))

            # scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            log.info('  epoch done in %1.2f sec.' % (time()-tstart))

        return cumulative_loss

    def evaluate_grad_auto(self, lpos):
        """Evaluate the gradient using automatic differentiation

        Args:
            lpos (torch.tensor): sampling points

        Returns:
            tuple: loss values and local energies
        """

        # compute the loss
        loss, eloc = self.loss(lpos)

        # add mo orthogonalization if required
        for reg in self.loss_reg:
            loss += reg()

        # compute local gradients
        self.opt.zero_grad()
        loss.backward()

        return loss, eloc

    def log_data(self):
        """Log basic information about the sampler."""

        log.info('')
        log.info(' QMC Solver ')

        if self.wf is not None:
            log.info(
                '  WaveFunction        : {0}', self.wf.__class__.__name__)
            for x in self.wf.__repr__().split('\n'):
                log.debug('   ' + x)

        if self.sampler is not None:
            log.info(
                '  Sampler             : {0}', self.sampler.__class__.__name__)
            for x in self.sampler.__repr__().split('\n'):
                log.debug('   ' + x)

        if self.opt is not None:
            log.info(
                '  Optimizer           : {0}', self.opt.__class__.__name__)
            for x in self.opt.__repr__().split('\n'):
                log.debug('   ' + x)

        if self.scheduler is not None:
            log.info(
                '  Scheduler           : {0}', self.scheduler.__class__.__name__)
            for x in self.scheduler.__repr__().split('\n'):
                log.debug('   ' + x)

    def log_data_opt(self, nepoch):
        """Log data for the optimization."""
        log.info('')
        log.info('  Optimization')
        log.info(
            '  Number Parameters   : {0}', self.wf.get_number_parameters())
        log.info('  Number of epoch     : {0}', nepoch)
        log.info(
            '  Batch size          : {0}', self.sampler.get_sampling_size())
        log.info('  Loss function       : {0}', self.loss.method)
        log.info('  Clip Loss           : {0}', self.loss.clip)
        log.info('  Gradients           : {0}', self.grad_method)
        log.info(
            '  Resampling mode     : {0}', self.resampler.options.mode)
        log.info(
            '  Resampling every    : {0}', self.resampler.options.resample_every)
        log.info(
            '  Resampling steps    : {0}', self.resampler.options.nstep_update)
        log.info(
            '  Output file         : {0}', self.hdf5file)
        log.info(
            '  Checkpoint every    : {0}', self.chkpt_every)
        log.info('')
