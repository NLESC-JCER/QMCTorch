import torch
from torch.utils.data import DataLoader

from deepqmc.solver.solver_base import SolverBase
from deepqmc.utils.torch_utils import (DataSet, Loss, OrthoReg)


class SolverOrbital(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None,
                 scheduler=None):
        """Serial solver

        Keyword Arguments:
            wf {WaveFunction} -- WaveFuntion object (default: {None})
            sampler {SamplerBase} -- Samppler (default: {None})
            optimizer {torch.optim} -- Optimizer (default: {None})
            scheduler (torch.schedul) -- Scheduler (default: {None})
        """

        SolverBase.__init__(self, wf, sampler, optimizer)

    def run(self, nepoch, batchsize=None, loss='variance',
            clip_loss=False, grad='auto'):
        """Run the optimization

        Arguments:
            nepoch {int} -- number of epoch

        Keyword Arguments:
            batchsize {int} -- batchsize. If None all the points at once (default: {None})
            loss {str} -- loss to be used  (default: {'variance'})
                          (energy, variance, weighted-energy, weighted-variance)
            clip_loss {bool} -- Remove points above/below 5 sigma of the mean (default: {False})
            grad {str} -- Method to compute the gradient (auto, manual) (default: {'auto'})
        """

        if 'lpos_needed' not in self.opt.__dict__.keys():
            self.opt.lpos_needed = False

        # sample the wave function
        pos = self.sample(ntherm=self.initial_sample.ntherm,
                          ndecor=self.initial_sample.ndecor)

        # resize the number of walkers
        _nwalker_save = self.sampler.walkers.nwalkers
        if self.resample.resample_from_last:
            self.sampler.walkers.nwalkers = pos.shape[0]
            self.sampler.nwalkers = pos.shape[0]

        # handle the batch size
        if batchsize is None:
            batchsize = len(pos)

        # change the number of steps
        _nstep_save = self.sampler.nstep
        _step_size_save = self.sampler.step_size

        self.sampler.nstep = self.resample.resample
        self.sampler.step_size = self.resample.step_size

        # create the data loader
        self.dataset = DataSet(pos)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batchsize)

        # get the loss
        self.loss = Loss(self.wf, method=loss, clip=clip_loss)

        # orthogonalization penalty for the MO coeffs
        self.ortho_loss = OrthoReg()

        cumulative_loss = []
        min_loss = 1E3

        # get the initial observalbe
        self.get_observable(self.obs_dict, pos)

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
                loss, eloc = self.evaluate_gradient(grad, lpos)
                cumulative_loss += loss

                # optimize the parameters
                self.optimization_step(lpos)

                # observable
                self.get_observable(self.obs_dict, pos,
                                    local_energy=eloc, ibatch=ibatch)

            # save the model if necessary
            if cumulative_loss < min_loss:
                min_loss = self.save_checkpoint(
                    n, cumulative_loss, self.save_model)

            self.print_observable(cumulative_loss)

            print('----------------------------------------')

            # resample the data
            pos = self._resample(n, nepoch, pos)

            if self.task == 'geo_opt':
                self.wf.update_mo_coeffs()

            if self.scheduler is not None:
                self.scheduler.step()

        # restore the sampler number of step
        self.sampler.nstep = _nstep_save
        self.sampler.step_size = _step_size_save
        self.sampler.walkers.nwalkers = _nwalker_save
        self.sampler.nwalkers = _nwalker_save

    def evaluate_gradient(self, grad, lpos):
        """Evaluate the gradient

        Arguments:
            grad {str} -- method of the gradient (auto, manual)
            lpos {torch.tensor} -- positions of the walkers


        Returns:
            tuple -- (loss, local energy)
        """
        if grad == 'auto':
            loss, eloc = self._evaluate_grad_auto(lpos)

        elif grad == 'manual':
            loss, eloc = self._evaluate_grad_manual(lpos)
        else:
            raise ValueError('Gradient method should be auto or stab')

        if torch.isnan(loss):
            raise ValueError("Nans detected in the loss")

        return loss, eloc

    def _evaluate_grad_auto(self, lpos):
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

    def _evaluate_grad_manual(self, lpos):
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
            raise ValueError('Manual gradient only for energy min')
