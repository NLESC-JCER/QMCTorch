import torch
from torch.utils.data import DataLoader

from .solver_base import SolverBase
from qmctorch.utils import (
    DataSet, Loss, OrthoReg, dump_to_hdf5, add_group_attr)


class SolverOrbital(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None,
                 scheduler=None, output=None):
        """Serial solver

        Keyword Arguments:
            wf {WaveFunction} -- WaveFuntion object (default: {None})
            sampler {SamplerBase} -- Samppler (default: {None})
            optimizer {torch.optim} -- Optimizer (default: {None})
            scheduler (torch.scheduler) -- Scheduler (default: {None})
        """

        SolverBase.__init__(self, wf, sampler, optimizer, output)

    def run(self, nepoch, batchsize=None, loss='variance',
            clip_loss=False, grad='auto', hdf5_group=None):
        """Run the optimization

        Arguments:
            nepoch {int} -- number of epoch

        Keyword Arguments:
            batchsize {int} -- batchsize. If None all the points at once
                               (default: {None})
            loss {str} -- loss to be used  (default: {'variance'})
                          (energy, variance,
                           weighted-energy, weighted-variance)
            clip_loss {bool} -- Remove points above/below 5 sigma of the mean
                                (default: {False})
            grad {str} -- Method to compute the gradient (auto, manual)
                          (default: {'auto'})
        """

        # observalbe
        if not hasattr(self, 'observable'):
            self.track_observable(['local_energy'])

        self.evaluate_gradient = {
            'auto': self.evaluate_grad_auto,
            'manual': self.evaluate_grad_manual}[grad]

        if 'lpos_needed' not in self.opt.__dict__.keys():
            self.opt.lpos_needed = False

        # sample the wave function
        pos = self.sampler(self.wf.pdf)

        # get the loss
        self.loss = Loss(self.wf, method=loss, clip=clip_loss)
        self.loss.use_weight = (
            self.resampling_options.resample_every > 1)

        # orthogonalization penalty for the MO coeffs
        self.ortho_loss = OrthoReg()

        # handle the batch size
        if batchsize is None:
            batchsize = len(pos)

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

        # get the initial observalbe
        self.store_observable(pos)

        # loop over the epoch
        for n in range(nepoch):
            print('----------------------------------------')
            print('epoch %d' % n)

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

            print('----------------------------------------')

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

    def evaluate_grad_auto(self, lpos):
        """Evaluate the gradient using automatic diff of the required loss.

        Arguments:
            lpos {torch.tensor} -- positions of the walkers

        Returns:
            tuple -- (loss, local energy)
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
        """Evaluate the gradient using a low variance method

        Arguments:
            lpos {torch.tensor} -- positions of the walkers

        Returns:
            tuple -- (loss, local energy)
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
