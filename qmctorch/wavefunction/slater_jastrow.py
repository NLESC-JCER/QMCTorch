from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from torch import nn

from .. import log
from ..utils import register_extra_attributes
from .orbitals.atomic_orbitals import AtomicOrbitals
from .pooling.orbital_configurations import OrbitalConfigurations
from .pooling.slater_pooling import SlaterPooling
from .jastrows.jastrow import set_jastrow
from .slater_jastrow_base import SlaterJastrowBase
from .jastrows.jastrow import set_jastrow


class SlaterJastrow(SlaterJastrowBase):

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
            >>> wf = SlaterJastrow(mol, configs='cas(2,2)')
        """

        super(SlaterJastrow, self).__init__(mol, configs, kinetic,
                                            use_jastrow, jastrow_type, cuda, include_all_mo)

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

        if self.use_jastrow:
            return J * self.fc(x)

        else:
            return self.fc(x)

    def ao2mo(self, ao):
        return self.mo(self.mo_scf(ao))

    def pos2mo(self, x, derivative=0):
        """Get the values of MOs

        Arguments:
            x {torch.tensor} -- positions of the electrons [nbatch, nelec*ndim]

        Keyword Arguments:
            derivative {int} -- order of the derivative (default: {0})

        Returns:
            torch.tensor -- MO matrix [nbatch, nelec, nmo]
        """
        return self.mo(self.mo_scf(self.ao(x, derivative=derivative)))

    def kinetic_energy_jacobi(self, x, **kwargs):
        r"""Compute the value of the kinetic enery using the Jacobi Formula.
        C. Filippi, Simple Formalism for Efficient Derivatives .

        .. math::
             \\frac{K(R)}{\Psi(R)} = Tr(A^{-1} B_{kin})

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)

        Returns:
            torch.tensor: values of the kinetic energy at each sampling points
        """

        ao, dao, d2ao = self.ao(x, derivative=[0, 1, 2])
        mo = self.ao2mo(ao)
        bkin = self.get_kinetic_operator(x, ao, dao, d2ao, mo)

        kin = self.pool.operator(mo, bkin)
        psi = self.pool(mo)
        out = self.fc(kin * psi) / self.fc(psi)
        return out

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

        # compute the determinants
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
        bgrad = bgrad.permute(3, 0, 1, 2)
        bgrad = bgrad.repeat(self.nelec, 1, 1, 1)
        # bgrad = bgrad.repeat(2, 1, 1, 1)

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

    def get_kinetic_operator(self, x, ao, dao, d2ao,  mo):
        """Compute the Bkin matrix

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)
            mo (torch.tensor, optional): precomputed values of the MOs

        Returns:
            torch.tensor: matrix of the kinetic operator
        """

        bkin = self.ao2mo(d2ao)

        if self.use_jastrow:

            jast, djast, d2jast = self.jastrow(x,
                                               derivative=[0, 1, 2],
                                               jacobian=False)

            djast = djast.transpose(1, 2) / jast.unsqueeze(-1)
            d2jast = d2jast / jast

            dmo = self.ao2mo(dao.transpose(2, 3)).transpose(2, 3)

            djast_dmo = (djast.unsqueeze(2) * dmo).sum(-1)
            d2jast_mo = d2jast.unsqueeze(-1) * mo

            bkin = bkin + 2 * djast_dmo + d2jast_mo

        return -0.5 * bkin
