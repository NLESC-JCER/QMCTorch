

import numpy as np
import torch
from .slater_jastrow_base import SlaterJastrowBase

from .jastrows.elec_elec.kernels.pade_jastrow_kernel import PadeJastrowKernel
from .jastrows.elec_elec.jastrow_factor_electron_electron import JastrowFactorElectronElectron


class SlaterJastrow(SlaterJastrowBase):

    def __init__(self, mol, configs='ground_state',
                 kinetic='jacobi',
                 jastrow_kernel=PadeJastrowKernel,
                 jastrow_kernel_kwargs={},
                 cuda=False,
                 include_all_mo=True):
        """Slater Jastrow wave function with electron-electron Jastrow factor

        .. math::
            \\Psi(R_{at}, r) = J(r)\\sum_n c_n D^\\uparrow_n(r^\\uparrow)D^\\downarrow_n(r^\\downarrow)

        with

        .. math::
            J(r) = \\exp\\left( K_{ee}(r) \\right)

        with K, a kernel function depending only on the electron-eletron distances 

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
            cuda (bool, optional): turns GPU ON/OFF  Defaults to Fals   e.
            include_all_mo (bool, optional): include either all molecular orbitals or only the ones that are
                                             popualted in the configs. Defaults to False
        Examples::
            >>> from qmctorch.scf import Molecule
            >>> from qmctorch.wavefunction import SlaterJastrow
            >>> mol = Molecule('h2o.xyz', calculator='adf', basis = 'dzp')
            >>> wf = SlaterJastrow(mol, configs='cas(2,2)')
        """

        super().__init__(mol, configs, kinetic, cuda, include_all_mo)

        # process the Jastrow
        if jastrow_kernel is not None:

            self.use_jastrow = True
            self.jastrow_type = jastrow_kernel.__name__
            self.jastrow = JastrowFactorElectronElectron(
                self.mol.nup, self.mol.ndown, jastrow_kernel,
                kernel_kwargs=jastrow_kernel_kwargs, cuda=cuda)

            if self.cuda:
                self.jastrow = self.jastrow.to(self.device)

        self.log_data()

    def forward(self, x, ao=None):
        """computes the value of the wave function for the sampling points

        .. math::
            \\Psi(R) =  J(R) \\sum_{n} c_n  D^{u}_n(r^u) \\times D^{d}_n(r^d)

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
        """Compute the value of the kinetic enery using the Jacobi Formula.
        C. Filippi, Simple Formalism for Efficient Derivatives .

        .. math::
            \\frac{\Delta \\Psi(R)}{\\Psi(R)} = \\Psi(R)^{-1} \\sum_n c_n (\\frac{\Delta D_n^u}{D_n^u} + \\frac{\Delta D_n^d}{D_n^d}) D_n^u D_n^d

        We compute the laplacian of the determinants through the Jacobi formula

        .. math::
            \\frac{\\Delta \\det(A)}{\\det(A)} = Tr(A^{-1} \\Delta A)

        Here :math:`A = J(R) \\phi` and therefore :

        .. math::
            \\Delta A = (\\Delta J) D + 2 \\nabla J \\nabla D + (\\Delta D) J

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

    def gradients_jacobi(self, x, sum_grad=False, pdf=False):
        """Compute the gradients of the wave function (or density) using the Jacobi Formula
        C. Filippi, Simple Formalism for Efficient Derivatives.

        .. math::
             \\frac{K(R)}{\Psi(R)} = Tr(A^{-1} B_{grad})

        The gradients of the wave function

        .. math::
            \\Psi(R) = J(R) \\sum_n c_n D^{u}_n D^{d}_n = J(R) \\Sigma

        are computed following

        .. math::
            \\nabla \\Psi(R) = \\left( \\nabla J(R) \\right) \\Sigma + J(R) \\left(\\nabla \Sigma \\right)

        with

        .. math::

            \\nabla \\Sigma =  \\sum_n c_n (\\frac{\\nabla D^u_n}{D^u_n} + \\frac{\\nabla D^d_n}{D^d_n}) D^u_n D^d_n

        that we compute with the Jacobi formula as:

        .. math::

            \\nabla \\Sigma =  \\sum_n c_n (Tr( (D^u_n)^{-1} \\nabla D^u_n) + Tr( (D^d_n)^{-1} \\nabla D^d_n)) D^u_n D^d_n

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)
            pdf (bool, optional) : if true compute the grads of the density

        Returns:
            torch.tensor: values of the gradients wrt the walker pos at each sampling points
        """

        # compute the mo values
        mo = self.ao2mo(self.ao(x))

        # compute the gradient operator matrix
        grad_ao = self.ao(x, derivative=1, sum_grad=False)

        # compute the derivatives of the MOs
        dmo = self.ao2mo(grad_ao.transpose(2, 3)).transpose(2, 3)
        dmo = dmo.permute(3, 0, 1, 2)

        # stride the tensor
        eye = torch.eye(self.nelec).to(self.device)
        dmo = dmo.unsqueeze(2) * eye.unsqueeze(-1)

        # reorder to have Nelec, Ndim, Nbatch, Nelec, Nmo
        dmo = dmo.permute(2, 0, 1, 3, 4)

        # flatten to have Nelec*Ndim, Nbatch, Nelec, Nmo
        dmo = dmo.reshape(-1, *(dmo.shape[2:]))

        # use the Jacobi formula to compute the value
        # the grad of each determinants and sum up the terms :
        # Tr( (D^u_n)^-1 \\nabla D^u_n) + Tr( (D^d_n)^-1 \\nabla D^d_n)
        grad_dets = self.pool.operator(mo, dmo)

        # compute the determinants
        # D^u_n D^d_n
        dets = self.pool(mo)

        # assemble the final values of \nabla \Sigma
        # \\sum_n c_n (Tr( (D^u_n)^-1 \\nabla D^u_n) + Tr( (D^d_n)^-1 \\nabla D^d_n)) D^u_n D^d_n
        out = self.fc(grad_dets * dets)
        out = out.transpose(0, 1).squeeze()

        if self.use_jastrow:

            nbatch = x.shape[0]

            # nbatch x 1
            jast = self.jastrow(x)

            # nbatch x ndim x nelec
            grad_jast = self.jastrow(x, derivative=1, sum_grad=False)

            # reorder grad_jast to nbtach x Nelec x Ndim
            grad_jast = grad_jast.permute(0, 2, 1)

            # compute J(R) (\nabla\Sigma)
            out = jast*out

            # add the product (\nabla J(R)) \Sigma
            out = out + \
                (grad_jast * self.fc(dets).unsqueeze(-1)).reshape(nbatch, -1)

        # compute the gradient of the pdf (i.e. the square of the wave function)
        # \nabla f^2 = 2 (\nabla f) f
        if pdf:
            out = 2 * out * self.fc(dets)
            if self.use_jastrow:
                out = out * jast

        return out

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
                                               sum_grad=False)

            djast = djast.transpose(1, 2) / jast.unsqueeze(-1)
            d2jast = d2jast / jast

            dmo = self.ao2mo(dao.transpose(2, 3)).transpose(2, 3)

            djast_dmo = (djast.unsqueeze(2) * dmo).sum(-1)
            d2jast_mo = d2jast.unsqueeze(-1) * mo

            bkin = bkin + 2 * djast_dmo + d2jast_mo

        return -0.5 * bkin
