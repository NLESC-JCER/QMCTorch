import torch
from torch.utils.data import DataLoader
from time import time
from copy import deepcopy
from .solver_base import SolverBase
from qmctorch.utils import (
    DataSet, Loss, dump_to_hdf5, add_group_attr)
from .. import log
from ..utils import save_trajectory


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

        # set which parameter to optimize
        self.configure_parameters(freeze=None)

        # how to compute the grad of the parameters
        self.configure_gradients('manual')

        # loss to use
        self.configure_loss(loss='energy', clip=False, ortho_mo=False)

    def configure_loss(self, loss='energy', clip=False, ortho_mo=False):
        """[summary]

        Args:
            loss(str, optional): merhod to compute the loss: variance or energy.
                                  Defaults to 'energy'.
            ortho_mo (bool, optional): apply regularization to orthogonalize the MOs.
                                       Defaults to False.
            clip (bool, optional): Clip the loss values at +/- 5std.
                                        Defaults to False.
        """

        # get the loss
        self.loss = Loss(self.wf, method=loss, clip=clip)
        self.loss.use_weight = (
            self.resampler.options.resample_every > 1)

        # orthogonalization penalty for the MO coeffs
        if ortho_mo:
            self.loss_reg.append(OrthoReg(self.wf.mo.weight))

    def configure_parameters(self, freeze=None):
        """Configure which parmeters to freeze

        Args:
            freeze (list, optional): list of parameters to freeze. Defaults to None.
        """

        # set the parameters we want to optimize/freeze
        self.set_params_requires_grad()
        self.freeze_params_list = freeze
        self.freeze_parameters(freeze)

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

    def configure_gradients(self, grad):
        """Configure how the grad of the parameters are evaluated

        Args:
            grad (str): 'auto' or 'manual'
        """
        if grad not in ['auto', 'manual']:
            raise ValueError('grad must be auto or manual')

        self.grad_method = grad
        self.evaluate_gradient = {
            'auto': self.evaluate_grad_auto,
            'manual': self.evaluate_grad_manual}[grad]

    def evaluate_grad_manual(self, lpos):
        """Evaluate the gradient using low variance express

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

        if self.loss.method in ['energy', 'weighted-energy']:

            ''' Get the gradient of the total energy
            dE/dk = < (dpsi/dk)/psi (E_L - <E_L >) >
            '''

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
