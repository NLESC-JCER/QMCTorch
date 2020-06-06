import torch
from torch.utils.data import DataLoader

from .solver_base import SolverBase
from qmctorch.utils import (
    DataSet, Loss, OrthoReg, dump_to_hdf5, add_group_attr)
from .. import log


class SolverOrbital(SolverBase):

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

        self.configure('wf_opt')
        self.ortho_mo = False

    def run(self, nepoch, batchsize=None, loss='energy',
            clip_loss=False, grad='manual', hdf5_group=None):
        """Run the optimization

        Args:
            nepoch (int): Number of optimziation step
            batchsize (int, optional): Number of sample in a mini batch.
                                       If None, all samples are used.
                                       Defaults to None.
            loss (str, optional): merhod to compute the loss: variance or energy.
                                  Defaults to 'energy'.
            clip_loss (bool, optional): Clip the loss values at +/- 5std.
                                        Defaults to False.
            grad (str, optional): method to compute the gradients: 'auto' or 'manual'.
                                  Defaults to 'auto'.
            hdf5_group (str, optional): name of the hdf5 group where to store the data.
                                        Defaults to wf.task.
        """

        log.info('')
        log.info('  Optimization')

        # observalbe
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

        # orthogonalization penalty for the MO coeffs
        self.ortho_loss = OrthoReg()

        # log data
        self.log_data_opt(nepoch, batchsize, loss, grad)

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

            log.info('')

            # resample the data
            pos = self.resample(n, pos)

            if self.task == 'geo_opt':
                self.wf.update_mo_coeffs()

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

        # add mo orthogonalization if required
        if self.wf.mo.weight.requires_grad and self.ortho_mo:
            loss += self.ortho_loss(self.wf.mo.weight)

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
            dE/dk = < (dpsi/dk)/psi (E_L - <E_L >) >
            '''

            # compute local energy and wf values
            _, eloc = self.loss(lpos, no_grad=True)
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

    def log_data_opt(self, nepoch, batchsize, loss, grad):
        """Log data for the optimization."""
        log.info('  Task                :', self.task)
        log.info(
            '  Number Parameters   : {0}', self.wf.get_number_parameters())
        log.info('  Number of epoch     : {0}', nepoch)
        log.info(
            '  Batch size          : {0}', self.sampler.get_sampling_size())
        log.info('  Loss function       : {0}', self.loss.method)
        log.info('  Clip Loss           : {0}', self.loss.clip)
        log.info('  Gradients           : {0}', grad)
        log.info(
            '  Resampling mode     : {0}', self.resampling_options.mode)
        log.info(
            '  Resampling every    : {0}', self.resampling_options.resample_every)
        log.info(
            '  Resampling steps    : {0}', self.resampling_options.nstep_update)
        log.info('')

    def configure(self, task='wf_opt', freeze=None):
        """Configure the optimization.

        Args:
            task (str, optional): Optimization task: 'wf_opt', 'geo_opt', 'fermi_opt'.
                                  Defaults to 'wf_opt'.
            freeze (list, optional): list pf layers to freeze.
                                     Defaults to None.
        """

        self.task = task

        if task == 'geo_opt':
            self.configure_geo_opt()

        elif task == 'wf_opt':
            self.configure_wf_opt()
            self.freeze_parameters(freeze)

        else:
            raise ValueError(
                'Configure can be wf_opt or geo_opt. Got ', task)

    def configure_geo_opt(self):
        """Configure the solver for geometry optimization."""

        # opt atom coordinate
        self.wf.ao.atom_coords.requires_grad = True

        # no ao opt
        self.wf.ao.bas_coeffs.requires_grad = False
        self.wf.ao.bas_exp.requires_grad = False

        # no jastrow opt
        self.wf.jastrow.weight.requires_grad = False

        # no mo opt
        for param in self.wf.mo.parameters():
            param.requires_grad = False

        # no ci opt
        self.wf.fc.weight.requires_grad = False

    def configure_wf_opt(self):
        """Configure the solver for wf optimization."""

        # opt all wf parameters
        self.wf.ao.bas_exp.requires_grad = True
        self.wf.ao.bas_coeffs.requires_grad = True
        for param in self.wf.mo.parameters():
            param.requires_grad = True
        self.wf.fc.weight.requires_grad = True
        self.wf.jastrow.weight.requires_grad = True

        # no opt the atom positions
        self.wf.ao.atom_coords.requires_grad = False

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
                    self.wf.jastrow.weight.requires_grad = False

                else:
                    opt_freeze = ['ci', 'mo', 'ao', 'jastrow']
                    raise ValueError(
                        'Valid arguments for freeze are :', opt_freeze)
