from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from torch import nn
import operator

from .. import log
from ..utils import register_extra_attributes
from .orbitals.atomic_orbitals_backflow import AtomicOrbitalsBackFlow
from .pooling.kinetic_pooling import KineticPooling
from .pooling.orbital_configurations import OrbitalConfigurations
from .pooling.slater_pooling import SlaterPooling
from .slater_jastrow_base import SlaterJastrowBase
from .jastrows.jastrow import set_jastrow


class SlaterJastrowBackFlow(SlaterJastrowBase):

    def __init__(self, mol, configs='ground_state',
                 kinetic='jacobi',
                 use_jastrow=True,
                 jastrow_type='pade_jastrow',
                 cuda=False,
                 include_all_mo=True,
                 **kwargs):
        """Implementation of the QMC Network.

        Args:
            mol (qmc.wavefunction.Molecule): a molecule object
            configs (str, optional): defines the CI configurations to be used. Defaults to 'ground_state'.
            kinetic (str, optional): method to compute the kinetic energy. Defaults to 'jacobi'.
            use_jastrow (bool, optional): turn jastrow factor ON/OFF. Defaults to True.
            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
            include_all_mo (bool, optional): include either all molecular orbitals or only the ones that are
                                             popualted in the configs. Defaults to False
        Examples::
            >>> mol = Molecule('h2o.xyz', calculator='adf', basis = 'dzp')
            >>> wf = SlaterJastrow(mol, configs='cas(2,2)')
        """

        super().__init__(mol, configs, kinetic,
                         use_jastrow, jastrow_type,
                         cuda, include_all_mo)

        self.ao = AtomicOrbitalsBackFlow(mol, cuda)

        self.jastrow = set_jastrow(
            jastrow_type, self.mol.nup, self.mol.ndown, self.cuda)

        if self.cuda:
            self.jastrow = self.jastrow.to(self.device)

        self.log_data()

    def forward(self, x, ao=None):
        """computes the value of the wave function for the sampling points

        .. math::
            \\Psi(R) = \\sum_{n} c_n D^{u}_n(r^u) \\times D^{d}_n(r^d)

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)
            ao (torch.tensor, optional): values of the atomic orbitals (Nbatch, Nelec, Nao)

        Returns:
            torch.tensor: values of the wave functions at each sampling point (Nbatch, 1)

        Examples::
            >>> mol = Molecule('h2.xyz', calculator='adf', basis = 'dzp')
            >>> wf = SlaterJastrow(mol, configs='cas(2,2)')
            >>> pos = torch.rand(500,6)
            >>> vals = wf(pos)
        """

        # compute the jastrow from the pos
        if self.use_jastrow:
            J = self.jastrow(x)

        # atomic orbital
        if ao is None:
            x = self.ao(x)
        else:
            x = ao

        # molecular orbitals
        x = self.mo_scf(x)

        # mix the mos
        x = self.mo(x)

        # pool the mos
        x = self.pool(x)

        # compute the CI and return
        if self.use_jastrow:
            return J * self.fc(x)

        else:
            return self.fc(x)

    def ao2mo(self, ao):
        return self.mo(self.mo_scf(ao))

    def pos2mo(self, x, derivative=0, jacobian=True):
        """Compute the MO vals from the pos

        Args:
            x ([type]): [description]
            derivative (int, optional): [description]. Defaults to 0.
            jacobian (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """

        ao = self.ao(x, derivative=derivative, jacobian=jacobian)
        return self.ao2mo(ao)

    def kinetic_energy_jacobi(self, x,  **kwargs):
        r"""Compute the value of the kinetic enery using the Jacobi Formula.
        C. Filippi, Simple Formalism for Efficient Derivatives .

        .. math::
             \\frac{K \Psi(R)}{\Psi(R)} = Tr(A^{-1} B_{kin})

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)
            kinpool (bool, optional): use kinetic pooling (deprecated). Defaults to False

        Returns:
            torch.tensor: values of the kinetic energy at each sampling points
        """

        # get ao values
        ao, dao, d2ao = self.ao(x, derivative=[0, 1, 2],
                                jacobian=False)

        # get the mo values
        mo = self.ao2mo(ao)
        dmo = self.ao2mo(dao)
        d2mo = self.ao2mo(d2ao)

        print('dmo', dmo.shape)
        print('d2mo', d2mo.shape)
        d2mo = d2mo.permute(2, 1, 0, 3)

        dmo = dmo.reshape(4, 3, 5, 4, 6)
        dmo = dmo.permute(3, 1, 2, 0, 4)
        dmo = dmo.reshape(12, 5, 4, 6)

        # compute the value of the slater det
        slater_dets = self.pool(mo)
        sum_slater_dets = self.fc(slater_dets)

        # compute ( tr(A_u^-1\Delta A_u) + tr(A_d^-1\Delta A_d) )
        hess = self.pool.operator(mo, d2mo)
        print('hess', hess.shape)

        # compute (tr(A_u^-1\nabla A_u) and tr(A_d^-1\nabla A_d))
        grad = self.pool.operator(mo, dmo, op=None)
        print('grad', grad[0].shape)

        # compute (tr((A_u^-1\nabla A_u)^2) + tr((A_d^-1\nabla A_d))^2)
        grad2 = self.pool.operator(mo, dmo, op_squared=True)
        print('grad2', grad2.shape)

        # assemble the total second derivative term
        hess = (hess.sum(0)
                + operator.add(*[(g**2).sum(0) for g in grad])
                - grad2.sum(0)
                + 2 * operator.mul(*grad).sum(0))

        hess = self.fc(hess * slater_dets) / sum_slater_dets

        if self.use_jastrow is False:
            print('exit')
            return -0.5 * hess

        # compute the Jastrow terms
        jast, djast, d2jast = self.jastrow(x,
                                           derivative=[0, 1, 2],
                                           jacobian=False)

        djast = djast / jast.unsqueeze(-1)
        djast = djast.permute(0, 2, 1).flatten(start_dim=1)

        d2jast = d2jast / jast

        grad_val = self.fc(operator.add(*grad) *
                           slater_dets) / sum_slater_dets
        grad_val = grad_val.squeeze().permute(1, 0)

        out = d2jast.sum(-1) + 2*(grad_val * djast).sum(-1) + \
            hess.squeeze().sum(0)

        return -0.5 * out

    def gradients_jacobi(self, x, jacobian=True):
        """Computes the gradients of the wf using Jacobi's Formula

        Args:
            x ([type]): [description]
        """

        # get the CMO matrix
        cmo = self.pos2cmo(x)

        # get the grad of the wf
        if jacobian:
            # bgrad = self.pos2cmo(x, derivative=1)
            bgrad = self.get_gradient_operator(x).sum(0)
        else:
            bgrad = self.get_gradient_operator(x)

        # compute the value of the grad using trace trick
        grad = self.pool.operator(cmo, bgrad, op=operator.add)

        # compute the total wf
        psi = self.pool(cmo)

        out = self.fc(grad * psi)
        out = out.transpose(0, 1)

        # assemble
        return out

    def get_hessian_operator(self, x, ao, dao, d2ao,  mo):
        """Compute the Bkin matrix

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)
            mo (torch.tensor, optional): precomputed values of the MOs

        Returns:
            torch.tensor: matrix of the kinetic operator
        """

        bhess = self.ao2mo(d2ao)

        if self.use_jastrow:

            jast, djast, d2jast = self.jastrow(x,
                                               derivative=[0, 1, 2],
                                               jacobian=False)

            djast = djast.transpose(1, 2) / jast.unsqueeze(-1)
            d2jast = d2jast / jast

            dmo = self.ao2mo(dao)

            djast_dmo = (djast.unsqueeze(2) * dmo).sum(-1)
            d2jast_mo = d2jast.unsqueeze(-1) * mo

            bhess = bhess + 2 * djast_dmo + d2jast_mo

        return bhess

    def get_gradient_operator(self, x):
        """Compute the gradient operator

        Args:
            x ([type]): [description]
            ao ([type]): [description]
            dao ([type]): [description]
        """

        mo = self.pos2mo(x)
        dmo = self.pos2mo(x, derivative=1, jacobian=False)

        jast = self.ordered_jastrow(x)
        djast = self.ordered_jastrow(x, derivative=1, jacobian=False)

        # reformat to have Nelec, Ndim, Nbatch, 1, Nmo
        djast = djast.permute(1, 3, 0, 2).unsqueeze(-2)

        # reformat to have Ndim, Nbatch, Nelec, Nmo
        dmo = dmo.permute(3, 0, 1, 2)

        # stride the tensor
        eye = torch.eye(self.nelec).to(self.device)
        dmo = dmo.unsqueeze(2) * eye.unsqueeze(-1)

        # reorder to have Nelec, Ndim, Nbatch, Nelec, Nmo
        dmo = dmo.permute(2, 0, 1, 3, 4)

        # assemble the derivative
        out = (mo * djast + dmo * jast)

        # collapse the first two dimensions
        out = out.reshape(-1, *(out.shape[2:]))
        return out
