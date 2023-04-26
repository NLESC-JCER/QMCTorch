from copy import deepcopy
from time import time

import torch
from torch.utils.data import DataLoader
from qmctorch.utils import (DataSet, Loss,
                            OrthoReg, add_group_attr,
                            dump_to_hdf5)

from .. import log
from .solver_base import SolverBase


class Solver(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None,
                 scheduler=None, output=None, rank=0):
        """Basic QMC solver

        Args:
            wf (qmctorch.WaveFunction, optional): wave function. Defaults to None.
            sampler (qmctorch.sampler, optional): Sampler. Defaults to None.
            optimizer (torch.optim, optional): optimizer. Defaults to None.
            scheduler (torch.optim, optional): scheduler. Defaults to None.
            output (str, optional): hdf5 filename. Defaults to None.
            rank (int, optional): rank of he process. Defaults to 0.
        """
        SolverBase.__init__(self, wf, sampler,
                            optimizer, scheduler, output, rank)

        self.set_params_requires_grad()

        self.configure(track=['local_energy'], freeze=None,
                       loss='energy', grad='manual',
                       ortho_mo=False, clip_loss=False,
                       resampling={'mode': 'update',
                                   'resample_every': 1,
                                   'nstep_update': 25})

    def configure(self, track=None, freeze=None,
                  loss=None, grad=None,
                  ortho_mo=None, clip_loss=False,
                  resampling=None):
        """Configure the solver

        Args:
            track (list, optional): list of observable to track. Defaults to ['local_energy'].
            freeze ([type], optional): list of parameters to freeze. Defaults to None.
            loss(str, optional): merhod to compute the loss: variance or energy.
                                  Defaults to 'energy'.
            grad (str, optional): method to compute the gradients: 'auto' or 'manual'.
                                  Defaults to 'auto'.
            ortho_mo (bool, optional): apply regularization to orthogonalize the MOs.
                                       Defaults to False.
            clip_loss (bool, optional): Clip the loss values at +/- X std. X defined in Loss
                                        as clip_num_std (default 5)
                                        Defaults to False.
        """

        # set the parameters we want to optimize/freeze
        self.set_params_requires_grad()
        self.freeze_params_list = freeze
        self.freeze_parameters(freeze)

        # track the observable we want
        if track is not None:
            self.track_observable(track)

        # define the grad calulation
        if grad is not None:
            self.grad_method = grad
            self.evaluate_gradient = {
                'auto': self.evaluate_grad_auto,
                'manual': self.evaluate_grad_manual}[grad]

        # resampling of the wave function
        if resampling is not None:
            self.configure_resampling(**resampling)

        # get the loss
        if loss is not None:
            self.loss = Loss(self.wf, method=loss, clip=clip_loss)
            self.loss.use_weight = (
                self.resampling_options.resample_every > 1)

        # orthogonalization penalty for the MO coeffs
        if ortho_mo is not None:
            self.ortho_mo = ortho_mo
            self.ortho_loss = OrthoReg()

    def set_params_requires_grad(self, wf_params=True, geo_params=False):
        """Configure parameters for wf opt."""

        # opt all wf parameters
        self.wf.ao.bas_exp.requires_grad = wf_params
        self.wf.ao.bas_coeffs.requires_grad = wf_params

        for param in self.wf.mo.parameters():
            param.requires_grad = wf_params

        self.wf.fc.weight.requires_grad = wf_params

        for param in self.wf.jastrow.parameters():
            param.requires_grad = wf_params

        # no opt the atom positions
        self.wf.ao.atom_coords.requires_grad = geo_params

    def freeze_parameters(self, freeze):
        """Freeze the optimization of specified params.

        Args:
            freeze (list): list of param to freeze
        """
        if freeze is not None:
            if not isinstance(freeze, list):
                freeze = [freeze]

            for name in freeze:
                if name.lower() == 'ci':
                    self.wf.fc.weight.requires_grad = False

                elif name.lower() == 'mo':
                    for param in self.wf.mo.parameters():
                        param.requires_grad = False

                elif name.lower() == 'ao':
                    self.wf.ao.bas_exp.requires_grad = False
                    self.wf.ao.bas_coeffs.requires_grad = False

                elif name.lower() == 'jastrow':
                    for param in self.wf.jastrow.parameters():
                        param.requires_grad = False

                else:
                    opt_freeze = ['ci', 'mo', 'ao', 'jastrow']
                    raise ValueError(
                        'Valid arguments for freeze are :', opt_freeze)

    def save_sampling_parameters(self, pos):
        """ save the sampling params."""
        self.sampler._nstep_save = self.sampler.nstep
        self.sampler._ntherm_save = self.sampler.ntherm
        self.sampler._nwalker_save = self.sampler.walkers.nwalkers

        if self.resampling_options.mode == 'update':
            self.sampler.ntherm = -1
            self.sampler.nstep = self.resampling_options.nstep_update
            self.sampler.walkers.nwalkers = pos.shape[0]
            self.sampler.nwalkers = pos.shape[0]

    def restore_sampling_parameters(self):
        """restore sampling params to their original values."""
        self.sampler.nstep = self.sampler._nstep_save
        self.sampler.ntherm = self.sampler._ntherm_save
        self.sampler.walkers.nwalkers = self.sampler._nwalker_save
        self.sampler.nwalkers = self.sampler._nwalker_save

    def geo_opt(self, nepoch, geo_lr=1e-2, batchsize=None,
                nepoch_wf_init=100, nepoch_wf_update=50,
                hdf5_group='geo_opt', chkpt_every=None, tqdm=False):
        """optimize the geometry of the molecule

        Args:
            nepoch (int): Number of optimziation step
            batchsize (int, optional): Number of sample in a mini batch.
                                       If None, all samples are used.
                                       Defaults to Never.
            hdf5_group (str, optional): name of the hdf5 group where to store the data.
                                        Defaults to 'geo_opt'.
            chkpt_every (int, optional): save a checkpoint every every iteration.
                                         Defaults to half the number of epoch
        """

        # save the optimizer used for the wf params
        opt_wf = deepcopy(self.opt)
        opt_wf.lpos_needed = self.opt.lpos_needed

        # create the optmizier for the geo opt
        opt_geo = torch.optim.SGD(self.wf.parameters(), lr=geo_lr)
        opt_geo.lpos_needed = False

        # save the grad method
        eval_grad_wf = self.evaluate_gradient

        # log data
        self.prepare_optimization(batchsize, None, tqdm)
        self.log_data_opt(nepoch, 'geometry optimization')

        # init the traj
        xyz = [self.wf.geometry(None)]

        # initial wf optimization
        self.set_params_requires_grad(wf_params=True,
                                      geo_params=False)
        self.freeze_parameters(self.freeze_params_list)
        self.run_epochs(nepoch_wf_init)

        # iterations over geo optim
        for n in range(nepoch):

            # make one step geo optim
            self.set_params_requires_grad(wf_params=False,
                                          geo_params=True)
            self.opt = opt_geo
            self.evaluate_gradient = self.evaluate_grad_auto
            self.run_epochs(1)
            xyz.append(self.wf.geometry(None))

            # make a few wf optim
            self.set_params_requires_grad(wf_params=True,
                                          geo_params=False)
            self.freeze_parameters(self.freeze_params_list)
            self.opt = opt_wf
            self.evaluate_gradient = eval_grad_wf

            cumulative_loss = self.run_epochs(nepoch_wf_update)

            # save checkpoint file
            if chkpt_every is not None:
                if (n > 0) and (n % chkpt_every == 0):
                    self.save_checkpoint(n, cumulative_loss)

        # restore the sampler number of step
        self.restore_sampling_parameters()

        # dump
        self.observable.geometry = xyz
        self.save_data(hdf5_group)

        return self.observable

    def run(self, nepoch, batchsize=None,
            hdf5_group='wf_opt', chkpt_every=None, tqdm=False):
        """Run a wave function optimization

        Args:
            nepoch (int): Number of optimziation step
            batchsize (int, optional): Number of sample in a mini batch.
                                       If None, all samples are used.
                                       Defaults to Never.
            hdf5_group (str, optional): name of the hdf5 group where to store the data.
                                        Defaults to 'wf_opt'.
            chkpt_every (int, optional): save a checkpoint every every iteration.
                                         Defaults to half the number of epoch
        """

        # prepare the optimization
        self.prepare_optimization(batchsize, chkpt_every, tqdm)
        self.log_data_opt(nepoch, 'wave function optimization')

        # run the epochs
        self.run_epochs(nepoch)

        # restore the sampler number of step
        self.restore_sampling_parameters()

        # dump
        self.save_data(hdf5_group)

        return self.observable

    def prepare_optimization(self, batchsize, chkpt_every, tqdm=False):
        """Prepare the optimization process

        Args:
            batchsize (int or None): batchsize
            chkpt_every (int or none): save a chkpt file every
        """

        # sample the wave function
        pos = self.sampler(self.wf.pdf, with_tqdm=tqdm)

        # handle the batch size
        if batchsize is None:
            batchsize = len(pos)

        # change the number of steps/walker size
        self.save_sampling_parameters(pos)

        # create the data loader
        self.dataset = DataSet(pos)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batchsize)

        for ibatch, data in enumerate(self.dataloader):
            self.store_observable(data, ibatch=ibatch)

        # chkpt
        self.chkpt_every = chkpt_every

    def save_data(self, hdf5_group):
        """Save the data to hdf5.

        Args:
            hdf5_group (str): name of group in the hdf5 file
        """
        self.observable.models.last = dict(self.wf.state_dict())

        hdf5_group = dump_to_hdf5(
            self.observable, self.hdf5file, hdf5_group)

        add_group_attr(self.hdf5file, hdf5_group, {'type': 'opt'})

    def run_epochs(self, nepoch):
        """Run a certain number of epochs

        Args:
            nepoch (int): number of epoch to run
        """

        # init the loss in case we have nepoch=0
        cumulative_loss = 0

        # loop over the epoch
        for n in range(nepoch):

            tstart = time()
            log.info('')
            log.info('  epoch %d' % n)

            cumulative_loss = 0

            # loop over the batches
            for ibatch, data in enumerate(self.dataloader):

                # port data to device
                lpos = data.to(self.device)

                # get the gradient
                loss, eloc = self.evaluate_gradient(lpos)
                cumulative_loss += loss

                # check for nan
                if torch.isnan(eloc).any():
                    log.info('Error : Nan detected in local energy')
                    return cumulative_loss

                # optimize the parameters
                self.optimization_step(lpos)

                # observable
                self.store_observable(
                    lpos, local_energy=eloc, ibatch=ibatch)

            # save the model if necessary
            if n == 0 or cumulative_loss < min_loss:
                min_loss = cumulative_loss
                self.observable.models.best = dict(
                    self.wf.state_dict())

            # save checkpoint file
            if self.chkpt_every is not None:
                if (n > 0) and (n % self.chkpt_every == 0):
                    self.save_checkpoint(n, cumulative_loss)

            self.print_observable(cumulative_loss, verbose=False)

            # resample the data
            self.dataset.data = self.resample(n, self.dataset.data)

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
        if self.wf.mo.weight.requires_grad and self.ortho_mo:
            loss += self.ortho_loss(self.wf.mo.weight)

        # compute local gradients
        self.opt.zero_grad()
        loss.backward()

        return loss, eloc

    def evaluate_grad_manual(self, lpos):
        """Evaluate the gradient using low variance expression

        Args:
            lpos ([type]): [description]

        Args:
            lpos (torch.tensor): sampling points

        Returns:
            tuple: loss values and local energies
        """

        # determine if we need the grad of eloc
        no_grad_eloc = True
        if self.wf.kinetic_method == 'auto':
            no_grad_eloc = False

        if self.wf.jastrow.requires_autograd:
            no_grad_eloc = False

        if self.loss.method in ['energy', 'weighted-energy']:

            # Get the gradient of the total energy
            # dE/dk = < (dpsi/dk)/psi (E_L - <E_L >) >
            

            # compute local energy and wf values
            _, eloc = self.loss(lpos, no_grad=no_grad_eloc)
            psi = self.wf(lpos)
            norm = 1. / len(psi)

            # evaluate the prefactor of the grads
            weight = eloc.clone()
            weight -= torch.mean(eloc)
            weight /= psi
            weight *= 2.
            weight *= norm

            # compute the gradients
            self.opt.zero_grad()
            psi.backward(weight)

            return torch.mean(eloc), eloc

        else:
            raise ValueError(
                'Manual gradient only for energy minimization')

    def log_data_opt(self, nepoch, task):
        """Log data for the optimization."""
        log.info('')
        log.info('  Optimization')
        log.info('  Task                :', task)
        log.info(
            '  Number Parameters   : {0}', self.wf.get_number_parameters())
        log.info('  Number of epoch     : {0}', nepoch)
        log.info(
            '  Batch size          : {0}', self.sampler.get_sampling_size())
        log.info('  Loss function       : {0}', self.loss.method)
        log.info('  Clip Loss           : {0}', self.loss.clip)
        log.info('  Gradients           : {0}', self.grad_method)
        log.info(
            '  Resampling mode     : {0}', self.resampling_options.mode)
        log.info(
            '  Resampling every    : {0}', self.resampling_options.resample_every)
        log.info(
            '  Resampling steps    : {0}', self.resampling_options.nstep_update)
        log.info(
            '  Output file         : {0}', self.hdf5file)
        log.info(
            '  Checkpoint every    : {0}', self.chkpt_every)
        log.info('')
