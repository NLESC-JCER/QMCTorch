

import torch

from torch import nn
import operator

from .. import log

from .orbitals.atomic_orbitals_backflow import AtomicOrbitalsBackFlow
from .orbitals.atomic_orbitals_orbital_dependent_backflow import AtomicOrbitalsOrbitalDependentBackFlow
from .slater_jastrow_base import SlaterJastrowBase
from .orbitals.backflow.kernels import BackFlowKernelInverse
from .jastrows.elec_elec.kernels.pade_jastrow_kernel import PadeJastrowKernel
from .jastrows.elec_elec.jastrow_factor_electron_electron import JastrowFactorElectronElectron


class SlaterJastrowBackFlow(SlaterJastrowBase):

    def __init__(self, mol, configs='ground_state',
                 kinetic='jacobi',
                 jastrow_kernel=PadeJastrowKernel,
                 jastrow_kernel_kwargs={},
                 backflow_kernel=BackFlowKernelInverse,
                 backflow_kernel_kwargs={},
                 orbital_dependent_backflow=False,
                 cuda=False,
                 include_all_mo=True):
        """Slater Jastrow wave function with electron-electron Jastrow factor and backflow

        .. math::
            \\Psi(R_{at}, r) = J(r)\\sum_n c_n D^\\uparrow_n(q^\\uparrow)D^\\downarrow_n(q^\\downarrow)

        with

        .. math::
            J(r) = \\exp\\left( K_{ee}(r) \\right)
        
        with K, a kernel function depending only on the electron-eletron distances, and

        .. math::
            q(r_i) = r_i + \\sum){j\\neq i} K_{BF}(r_{ij})(r_i-r_j)

        is a backflow transformation defined by the kernel K_{BF}. Note that different transformation
        can be used for different orbital via the `orbital_dependent_backflow` option.

        Args:
        Args:
            mol (Molecule): a QMCTorch molecule object
            configs (str, optional): defines the CI configurations to be used. Defaults to 'ground_state'.
                - ground_state : only the ground state determinant in the wave function
                - single(n,m) : only single excitation with n electrons and m orbitals 
                - single_double(n,m) : single and double excitation with n electrons and m orbitals
                - cas(n, m) : all possible configuration using n eletrons and m orbitals                   
            kinetic (str, optional): method to compute the kinetic energy. Defaults to 'jacobi'.
                - jacobi : use the Jacobi formula to compute the kinetic energy 
                - auto : use automatic differentiation to compute the kinetic energy
            jastrow_kernel (JastrowKernelBase, optional) : Class that computes the jastrow kernels
            jastrow_kernel_kwargs (dict, optional) : keyword arguments for the jastrow kernel contructor
            backflow_kernel (BackFlowKernelBase, optional) : kernel function of the backflow transformation. 
                - By default an inverse kernel K(r_{ij}) = w/r_{ij} is used
            backflow_kernel_kwargs (dict, optional) : keyword arguments for the backflow kernel contructor
            orbital_dependent_backflow (bool, optional) : every orbital has a different transformation if True. Default to False 
            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
            include_all_mo (bool, optional): include either all molecular orbitals or only the ones that are
                                             popualted in the configs. Defaults to False
    
        Examples::
            >>> from qmctorch.scf import Molecule
            >>> from qmctorch.wavefunction import SlaterJastrowBackFlow
            >>> mol = Molecule('h2o.xyz', calculator='adf', basis = 'dzp')
            >>> wf = SlaterJastrowBackFlow(mol, configs='cas(2,2)')
        """

        super().__init__(mol, configs, kinetic, cuda, include_all_mo)

        # process the backflow transformation
        if orbital_dependent_backflow:
            self.ao = AtomicOrbitalsOrbitalDependentBackFlow(
                mol, backflow_kernel, backflow_kernel_kwargs, cuda)
        else:
            self.ao = AtomicOrbitalsBackFlow(
                mol, backflow_kernel, backflow_kernel_kwargs, cuda)

        # process the Jastrow
        self.jastrow = JastrowFactorElectronElectron(
            self.mol.nup, self.mol.ndown, jastrow_kernel,
            kernel_kwargs=jastrow_kernel_kwargs, cuda=cuda)

        if jastrow_kernel is not None:
            self.use_jastrow = True
            self.jastrow_type = jastrow_kernel.__name__

        if self.cuda:
            self.jastrow = self.jastrow.to(self.device)
            self.ao = self.ao.to(self.device)

        self.log_data()

    def forward(self, x, ao=None):
        """computes the value of the wave function for the sampling points

        .. math::
            J(R) \\Psi(R) =  J(R) \\sum_{n} c_n D^{u}_n(r^u) \\times D^{d}_n(r^d)

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
        """transforms AO values in to MO values."""
        return self.mo(self.mo_scf(ao))

    def pos2mo(self, x, derivative=0, sum_grad=True):
        """Compute the MO vals from the pos

        Args:
            x ([type]): [description]
            derivative (int, optional): [description]. Defaults to 0.
            sum_grad (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """

        ao = self.ao(x, derivative=derivative, sum_grad=sum_grad)
        return self.ao2mo(ao)

    def kinetic_energy_jacobi(self, x,  **kwargs):
        """Compute the value of the kinetic enery using the Jacobi Formula.


        .. math::
             \\frac{\\Delta (J(R) \\Psi(R))}{ J(R) \\Psi(R)} = \\frac{\\Delta J(R)}{J(R)}
                                                          + 2 \\frac{\\nabla J(R)}{J(R)} \\frac{\\nabla \\Psi(R)}{\\Psi(R)}
                                                          + \\frac{\\Delta \\Psi(R)}{\\Psi(R)}

        The lapacian of the determinental part is computed via

        .. math::
            \\Delta_i \\Psi(R) \\sum_n c_n ( \\frac{\\Delta_i D_n^{u}}{D_n^{u}} +
                                           \\frac{\\Delta_i D_n^{d}}{D_n^{d}} +
                                           2 \\frac{\\nabla_i D_n^{u}}{D_n^{u}} \\frac{\\nabla_i D_n^{d}}{D_n^{d}} )
                                           D_n^{u} D_n^{d}

        Since the backflow orbitals are multi-electronic the laplacian of the determinants
        are obtained

        .. math::
            \\frac{\\Delta det(A)}{det(A)} = Tr(A^{-1} \\Delta A) +
                                             Tr(A^{-1} \\nabla A) Tr(A^{-1} \\nabla A) +
                                             Tr( (A^{-1} \\nabla A) (A^{-1} \\nabla A ))


        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)

        Returns:
            torch.tensor: values of the kinetic energy at each sampling points
        """

        # get ao values
        ao, dao, d2ao = self.ao(
            x, derivative=[0, 1, 2], sum_grad=False)

        # get the mo values
        mo = self.ao2mo(ao)
        dmo = self.ao2mo(dao)
        d2mo = self.ao2mo(d2ao)

        # compute the value of the slater det
        slater_dets = self.pool(mo)
        sum_slater_dets = self.fc(slater_dets)

        # compute ( tr(A_u^-1\Delta A_u) + tr(A_d^-1\Delta A_d) )
        hess = self.pool.operator(mo, d2mo)

        # compute (tr(A_u^-1\nabla A_u) and tr(A_d^-1\nabla A_d))
        grad = self.pool.operator(mo, dmo, op=None)

        # compute (tr((A_u^-1\nabla A_u)^2) + tr((A_d^-1\nabla A_d))^2)
        grad2 = self.pool.operator(mo, dmo, op_squared=True)

        # assemble the total second derivative term
        hess = (hess.sum(0)
                + operator.add(*[(g**2).sum(0) for g in grad])
                - grad2.sum(0)
                + 2 * operator.mul(*grad).sum(0))

        hess = self.fc(hess * slater_dets) / sum_slater_dets

        if self.use_jastrow is False:
            return -0.5 * hess

        # compute the Jastrow terms
        jast, djast, d2jast = self.jastrow(x,
                                           derivative=[0, 1, 2],
                                           sum_grad=False)

        # prepare the second derivative term d2Jast/Jast
        # Nbatch x Nelec
        d2jast = d2jast / jast

        # prepare the first derivative term
        djast = djast / jast.unsqueeze(-1)

        # -> Nelec x Ndim x Nbatch
        djast = djast.permute(2, 1, 0)

        # -> [Nelec*Ndim] x Nbatch
        djast = djast.reshape(-1, djast.shape[-1])

        # prepare the grad of the dets
        # [Nelec*Ndim] x Nbatch x 1
        grad_val = self.fc(operator.add(*grad) *
                           slater_dets) / sum_slater_dets

        # [Nelec*Ndim] x Nbatch
        grad_val = grad_val.squeeze()

        # assemble the derivaite terms
        out = d2jast.sum(-1) + 2*(grad_val * djast).sum(0) + \
            hess.squeeze(-1)

        return -0.5 * out.unsqueeze(-1)

    def gradients_jacobi(self, x, sum_grad=True):
        """Computes the gradients of the wf using Jacobi's Formula

        Args:
            x ([type]): [description]
        """
        raise NotImplementedError(
            'Gradient through Jacobi formulat not implemented for backflow orbitals')
