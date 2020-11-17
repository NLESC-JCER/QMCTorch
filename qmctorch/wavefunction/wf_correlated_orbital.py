from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from torch import nn

from .. import log
from ..utils import register_extra_attributes
from .orbitals.atomic_orbitals import AtomicOrbitals
from .pooling.kinetic_pooling import KineticPooling
from .pooling.orbital_configurations import OrbitalConfigurations
from .pooling.slater_pooling import SlaterPooling
from .jastrows.jastrow import set_jastrow
from .wf_orbital_base import OrbitalBase
from .jastrows.jastrow import set_jastrow_correlated


class CorrelatedOrbital(OrbitalBase):

    def __init__(self, mol, configs='ground_state',
                 kinetic='jacobi',
                 use_jastrow=True,
                 jastrow_type='pade_jastrow',
                 cuda=False,
                 include_all_mo=True):
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
            >>> wf = Orbital(mol, configs='cas(2,2)')
        """

        if use_jastrow is False:
            raise ValueError('use_jastrow = False is invalid for CorrelatedOrbital wave functions, \
                              use Orbital wave function if you do not want to use Jastrow factors')

        super(CorrelatedOrbital, self).__init__(mol, configs,
                                                kinetic, use_jastrow, jastrow_type,
                                                cuda, include_all_mo)

        self.jastrow_fn = set_jastrow_correlated(
            jastrow_type, self.mol.nup, self.mol.ndown, self.nmo_opt, self.cuda)

    def jastrow(self, pos, derivative=0, jacobian=True):
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
        jast_vals = self.jastrow_fn(pos, derivative, jacobian)

        def permute(vals):
            """transpose the data depending on it number of dim."""
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
            >>> wf = Orbital(mol, configs='cas(2,2)')
            >>> pos = torch.rand(500,6)
            >>> vals = wf(pos)
        """

        # compute the jastrow from the pos
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

    def pos2mo(self, x, derivative=0):
        ao = self.ao(x, derivative=derivative)
        return self.ao2mo(ao)

    def pos2cmo(self, x, derivative=0):
        """Get the values of correlated MOs

        Arguments:
            x {torch.tensor} -- positions of the electrons [nbatch, nelec*ndim]


        Returns:
            torch.tensor -- MO matrix [nbatch, nelec, nmo]
        """
        if derivative == 0:
            mo = self.pos2mo(x)
            jast = self.jastrow(x)
            return jast * mo

        elif derivative == 1:

            mo = self.pos2mo(x)
            dmo = self.pos2mo(x, derivative=1)

            jast = self.jastrow(x)
            djast = self.jastrow(x, derivative=1)

            # return mo * djast.sum(1).unsqueeze(1) + jast * dmo
            return mo.sum(1).unsqueeze(1) * djast + jast * dmo

        elif derivative == 2:

            # atomic orbital
            ao, dao, d2ao = self.ao(x, derivative=[0, 1, 2])

            # bare molecular orbitals
            mo = self.ao2mo(ao)
            dmo = self.ao2mo(dao.transpose(2, 3)).transpose(2, 3)
            d2mo = self.ao2mo(d2ao)

            # jastrows
            jast, djast, d2jast = self.jastrow(x,
                                               derivative=[0, 1, 2],
                                               jacobian=False)

            # terms of the kin op
            jast_d2mo = d2mo * jast
            djast_dmo = (djast * dmo).sum(-1)
            # d2jast_mo = d2jast.sum(1).unsqueeze(1) * mo
            d2jast_mo = d2jast * mo.sum(1).unsqueeze(1)

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

        cmo = self.pos2cmo(x)
        bkin = self.get_kinetic_operator(x)

        kin = self.pool.operator(cmo, bkin)
        psi = self.pool(cmo)
        out = self.fc(kin * psi) / self.fc(psi)

        return out

    def get_kinetic_operator(self, x):
        """Compute the Bkin matrix

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)
            mo (torch.tensor, optional): precomputed values of the MOs

        Returns:
            torch.tensor: matrix of the kinetic operator
        """

        return -0.5 * self.pos2cmo(x, derivative=2)

    def gradients_jacobi(self, x, pdf=False):
        """Compute the gradients of the wave function (or density) using the Jacobi Formula
        C. Filippi, Simple Formalism for Efficient Derivatives.

        .. math::
             \\frac{K(R)}{\Psi(R)} = Tr(A^{-1} B_{grad})

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)
            pdf (bool, optional) : if true compute the grads of the density

        Returns:
            torch.tensor: values of the gradients wrt the walker pos at each sampling points
        """

        # compute the gradient operator matrix
        ao = self.ao(x)
        grad_ao = self.ao(x, derivative=1, jacobian=False)
        mo = self.ao2mo(ao)
        bgrad = self.get_grad_operator(x, ao, grad_ao, mo)

        # use the Jacobi formula to compute the value
        # the grad of each determinants
        grads = self.pool.operator(mo, bgrad)

        # comoute the determinants
        dets = self.pool(mo)

        # CI sum
        psi = self.fc(dets)

        # assemble the final values of
        # nabla psi / psi
        grads = self.fc(grads * dets)
        grads = grads.transpose(0, 1).squeeze()

        # multiply by psi to get the grads
        # grads = grads * psi
        if self.use_jastrow:
            jast = self.jastrow(x)
            grads = grads * jast

        # if we need the grads of the pdf
        if pdf:
            grads = 2*grads*psi
            if self.use_jastrow:
                grads = grads*jast

        return grads

    def get_grad_operator(self, x, ao, grad_ao, mo):
        """Compute the gradient operator

        Args:
            x ([type]): [description]
            ao ([type]): [description]
            dao ([type]): [description]
        """

        bgrad = self.ao2mo(grad_ao.transpose(2, 3)).transpose(2, 3)
        bgrad = bgrad.permute(3, 0, 1, 2).repeat(2, 1, 1, 1)

        for ielec in range(self.nelec):
            bgrad[ielec*3:(ielec+1)*3, :, :ielec, :] = 0
            bgrad[ielec*3:(ielec+1)*3, :, ielec+1:, :] = 0

        if self.use_jastrow:

            jast = self.jastrow(x)
            grad_jast = self.jastrow(x,
                                     derivative=1,
                                     jacobian=False)
            grad_jast = grad_jast.transpose(1, 2) / jast.unsqueeze(-1)

            grad_jast = grad_jast.flatten(start_dim=1)
            grad_jast = grad_jast.transpose(0, 1)

            grad_jast = grad_jast.unsqueeze(2).unsqueeze(3)

            bgrad = bgrad + 0.5 * grad_jast * mo.unsqueeze(0)

        return bgrad
