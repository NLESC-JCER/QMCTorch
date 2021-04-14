from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from torch import nn
import operator

from .. import log
from ..utils import register_extra_attributes
from .orbitals.atomic_orbitals import AtomicOrbitals
from .pooling.orbital_configurations import OrbitalConfigurations
from .pooling.slater_pooling import SlaterPooling
from .slater_jastrow_base import SlaterJastrowBase
from .jastrows.jastrow_correlated_orbitals import set_jastrow_correlated


class SlaterJastrowOrbital(SlaterJastrowBase):

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

        if use_jastrow is False:
            raise ValueError('use_jastrow = False is invalid for SlaterJastrowOrbital wave functions, \
                              use SlaterJastrow wave function if you do not want to use Jastrow factors')

        super().__init__(mol, configs,
                         kinetic, use_jastrow, jastrow_type,
                         cuda, include_all_mo)

        self.jastrow = set_jastrow_correlated(
            jastrow_type, self.mol.nup, self.mol.ndown, self.nmo_opt, self.cuda, **kwargs)

        if self.cuda:
            self.jastrow = self.jastrow.to(self.device)

        self.jastrow_type = self.jastrow.__class__.__name__

        self.log_data()

    def ordered_jastrow(self, pos, derivative=0, jacobian=True):
        """Returns the value of the jastrow with the correct dimensions

        Args:
            pos (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0,1,2,).
                            Defaults to 0.
            jacobian (bool, optional): Return the jacobian (i.e. the sum of
                                       the derivatives) or the individual
                                       terms. Defaults to True.
                                       False only for derivative=1

        Returns:
            torch.tensor: value of the jastrow parameter for all confs
                          Nbatch, Nelec, Nmo (jacobian = True)
                          Nbatch, Nelec, Nmo, Ndim (jacobian = False)
        """
        jast_vals = self.jastrow(pos, derivative, jacobian)

        def permute(vals):
            """transpose the data depending on the number of dim."""
            if vals.ndim == 3:
                return vals.permute(1, 2, 0)
            elif vals.ndim == 4:
                return vals.permute(1, 3, 0, 2)

        if isinstance(jast_vals, tuple):
            return tuple([permute(v) for v in jast_vals])
        else:
            return permute(jast_vals)

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
        J = self.ordered_jastrow(x)

        # atomic orbital
        if ao is None:
            x = self.ao(x)
        else:
            x = ao

        # molecular orbitals
        x = self.mo_scf(x)

        # mix the mos
        x = self.mo(x)

        # jastrow for each orbital
        x = J * x

        # pool the mos
        x = self.pool(x)

        # compute the CI and return
        return self.fc(x)

    def ao2mo(self, ao):
        return self.mo(self.mo_scf(ao))

    def ao2cmo(self, ao, jastrow):
        return jastrow * self.mo(self.mo_scf(ao))

    def pos2mo(self, x, derivative=0, jacobian=True):
        """Compute the uncorrelated MOs from the positions."""

        ao = self.ao(x, derivative=derivative, jacobian=jacobian)
        if jacobian:
            return self.ao2mo(ao)
        else:
            return self.ao2mo(ao.transpose(2, 3)).transpose(2, 3)

    def pos2cmo(self, x, derivative=0, jacobian=True):
        """Get the values of correlated MOs

        Arguments:
            x {torch.tensor} -- positions of the electrons [nbatch, nelec*ndim]


        Returns:
            torch.tensor -- MO matrix [nbatch, nelec, nmo]
        """
        if derivative == 0:
            mo = self.pos2mo(x)
            jast = self.ordered_jastrow(x)
            return jast * mo

        elif derivative == 1:

            mo = self.pos2mo(x)
            dmo = self.pos2mo(x, derivative=1, jacobian=jacobian)

            jast = self.ordered_jastrow(x)
            djast = self.ordered_jastrow(
                x, derivative=1, jacobian=jacobian)

            if jacobian:
                return mo * djast.sum(1).unsqueeze(1) + jast * dmo
            else:
                return mo.unsqueeze(-1) * djast.sum(1).unsqueeze(1) + jast.unsqueeze(-1) * dmo

        elif derivative == 2:

            # atomic orbital
            ao, dao, d2ao = self.ao(x, derivative=[0, 1, 2])

            # bare molecular orbitals
            mo = self.ao2mo(ao)
            dmo = self.ao2mo(dao.transpose(2, 3)).transpose(2, 3)
            d2mo = self.ao2mo(d2ao)

            # jastrows
            jast, djast, d2jast = self.ordered_jastrow(x,
                                                       derivative=[
                                                           0, 1, 2],
                                                       jacobian=False)
            # terms of the kin op
            jast_d2mo = d2mo * jast
            djast_dmo = (djast * dmo).sum(-1)
            d2jast_mo = d2jast.sum(1).unsqueeze(1) * mo

            # assemble kin op
            return jast_d2mo + 2 * djast_dmo + d2jast_mo

    def kinetic_energy_jacobi(self, x,  **kwargs):
        r"""Compute the value of the kinetic enery using the Jacobi Formula.
        C. Filippi, Simple Formalism for Efficient Derivatives .

        .. math::
             \\frac{K(R)}{\Psi(R)} = Tr(A^{-1} B_{kin})

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)
            kinpool (bool, optional): use kinetic pooling (deprecated). Defaults to False

        Returns:
            torch.tensor: values of the kinetic energy at each sampling points
        """

        # get the matrix of correlated orbitals for all elec
        cmo = self.pos2cmo(x)

        # compute the value of the slater det
        slater_dets = self.pool(cmo)

        # compute  \Delta A (A = matrix of the correlated MO)
        bhess = self.pos2cmo(x, 2)

        # compute ( tr(A_u^-1\Delta A_u) + tr(A_d^-1\Delta A_d) )
        hess = self.pool.operator(cmo, bhess)

        # compute \grad A
        bgrad = self.get_gradient_operator(x)

        # compute (tr(A_u^-1\nabla A_u) * tr(A_d^-1\nabla A_d))
        grad = self.pool.operator(cmo, bgrad, op=None)
        grad2 = self.pool.operator(cmo, bgrad, op_squared=True)

        # assemble the total kinetic values
        kin = - 0.5 * (hess
                       + operator.add(*[(g**2).sum(0) for g in grad])
                       - grad2.sum(0)
                       + 2 * operator.mul(*grad).sum(0))

        # assemble
        return self.fc(kin * slater_dets) / self.fc(slater_dets)

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

    def get_hessian_operator(self, x):
        """Compute the Bkin matrix

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)
            mo (torch.tensor, optional): precomputed values of the MOs

        Returns:
            torch.tensor: matrix of the kinetic operator
        """
        if 1:
            mo = self.pos2mo(x)
            dmo = self.pos2mo(x, derivative=1, jacobian=False)
            d2mo = self.pos2mo(x, derivative=2)

            jast = self.ordered_jastrow(x)
            djast = self.ordered_jastrow(
                x, derivative=1, jacobian=False)
            d2jast = self.ordered_jastrow(x, derivative=2)

            # \Delta_n J * MO
            d2jast_mo = d2jast.permute(1, 0, 2).unsqueeze(2) * mo

            # stride d2mo
            eye = torch.eye(self.nelec).to(self.device)
            d2mo = d2mo.unsqueeze(2) * eye.unsqueeze(-1)

            # reshape d2mo to nelec, nbatch, nelec, nmo
            d2mo = d2mo.permute(1, 0, 2, 3)

            # \Delta_n MO * J
            d2mo_jast = d2mo * jast.repeat(1, self.nelec, 1)

            # reformat to have Ndim, Nbatch, Nelec, Nmo
            dmo = dmo.permute(3, 0, 1, 2)

            # stride
            eye = torch.eye(self.nelec).to(self.device)
            dmo = dmo.unsqueeze(2) * eye.unsqueeze(-1)

            # reorder to have Nelec, Ndim, Nbatch, Nelec, Nmo
            dmo = dmo.permute(2, 0, 1, 3, 4)

            # reshape djast to Nelec, Ndim, Nbatch, 1, Nmo
            djast = djast.permute(1, 3, 0, 2).unsqueeze(-2)

            # \nabla jast \nabla mo
            djast_dmo = (djast * dmo)

            # sum over ndim -> Nelec, Nbatch, Nelec, Nmo
            djast_dmo = djast_dmo.sum(1)

            return d2mo_jast + d2jast_mo + 2*djast_dmo

        else:
            return self.pos2cmo(x, derivative=2)

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
