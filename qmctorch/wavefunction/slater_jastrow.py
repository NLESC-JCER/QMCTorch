import torch
from typing import Union, Optional, List
from typing_extensions import Self
from scipy.optimize import curve_fit
from copy import deepcopy
import numpy as np
from torch import nn
import operator
import matplotlib.pyplot as plt

from .. import log

from ..scf import Molecule
from .wf_base import WaveFunction
from .orbitals.backflow.backflow_transformation import BackFlowTransformation
from .jastrows.elec_elec.jastrow_factor_electron_electron import (
    JastrowFactorElectronElectron,
)
from .jastrows.elec_elec.kernels import PadeJastrowKernel
from .jastrows.combine_jastrow import CombineJastrow
from .orbitals.atomic_orbitals import AtomicOrbitals
from .orbitals.molecular_orbitals import MolecularOrbitals
from .orbitals.atomic_orbitals_backflow import AtomicOrbitalsBackFlow
from .pooling.slater_pooling import SlaterPooling
from .pooling.orbital_configurations import OrbitalConfigurations
from ..utils import register_extra_attributes
from ..utils.constants import BOHR2ANGS


class SlaterJastrow(WaveFunction):
    def __init__(
        self,
        mol: Molecule,
        jastrow: Optional[Union[str, nn.Module, None]] = "default",
        backflow: Optional[Union[BackFlowTransformation, None]] = None,
        configs: str = "ground_state",
        kinetic: str = "jacobi",
        cuda: bool = False,
        include_all_mo: bool = True,
        mix_mo: bool = False,
        orthogonalize_mo: bool = False,
    ) -> None:
        """Slater Jastrow wave function with electron-electron Jastrow factor

        .. math::
            \\Psi(R_{at}, r) = J(r)\\sum_n c_n D^\\uparrow_n(r^\\uparrow)D^\\downarrow_n(r^\\downarrow)

        with

        .. math::
            J(r) = \\exp\\left( K_{ee}(r) \\right)

        with K, a kernel function depending only on the electron-eletron distances

        Args:
            mol (Molecule): a QMCTorch molecule object
            jastrow (str, optional) : Class that computes the jastrow kernels. Defaults to 'default'.
            backflow (BackFlowKernelBase, optional) : kernel function of the backflow transformation. Defaults to None.
            configs (str, optional): defines the CI configurations to be used. Defaults to 'ground_state'.
                - ground_state : only the ground state determinant in the wave function
                - single(n,m) : only single excitation with n electrons and m orbitals
                - single_double(n,m) : single and double excitation with n electrons and m orbitals
                - cas(n, m) : all possible configuration using n eletrons and m orbitals
            kinetic (str, optional): method to compute the kinetic energy. Defaults to 'jacobi'.
                - jacobi : use the Jacobi formula to compute the kinetic energy
                - auto : use automatic differentiation to compute the kinetic energy
            cuda (bool, optional): turns GPU ON/OFF  Defaults to False..
            include_all_mo (bool, optional): include either all molecular orbitals or only the ones that are
                                             popualted in the configs. Defaults to False
            orthogonalize_mo (bool, optional): orthogonalize the molecular orbitals. Defaults to False
        Examples::
            >>> from qmctorch.scf import Molecule
            >>> from qmctorch.wavefunction import SlaterJastrow
            >>> mol = Molecule('h2o.xyz', calculator='adf', basis = 'dzp')
            >>> wf = SlaterJastrow(mol, configs='cas(2,2)')
        """

        super().__init__(mol.nelec, 3, kinetic, cuda)

        # check for cuda
        if not torch.cuda.is_available and self.cuda:
            raise ValueError("Cuda not available, use cuda=False")

        # check for conf/mo size
        if not include_all_mo and configs.startswith("cas("):
            raise ValueError("CAS calculation only possible with include_all_mo=True")

        # molecule/atoms
        self.mol = mol
        self.atoms = mol.atoms
        self.natom = mol.natom

        # electronic confs
        self.init_config(configs)

        # atomic orbitals init
        self.init_atomic_orb(backflow)

        # init mo layer
        self.init_molecular_orb(include_all_mo, mix_mo, orthogonalize_mo)

        # initialize the slater det calculator
        self.init_slater_det_calculator()

        # initialize the fully connected layer
        self.init_fc_layer()

        # init the jastrow
        self.init_jastrow(jastrow)

        # init the knientic calc methods
        self.init_kinetic(kinetic, backflow)

        # register the callable for hdf5 dump
        register_extra_attributes(self, ["ao", "mo", "jastrow", "pool", "fc"])

        self.log_data()

    def init_atomic_orb(self, backflow: Union[BackFlowTransformation, None]) -> None:
        """Initialize the atomic orbital layer."""
        # self.backflow = backflow
        if backflow is None:
            self.use_backflow = False
            self.ao = AtomicOrbitals(self.mol, self.cuda)
        else:
            self.use_backflow = True
            self.backflow_type = backflow.__repr__()
            self.ao = AtomicOrbitalsBackFlow(self.mol, backflow, self.cuda)

        if self.cuda:
            self.ao = self.ao.to(self.device)

    def init_molecular_orb(self, include_all_mo, mix_mo, orthogonalize_mo):
        """initialize the molecular orbital layers"""

        if orthogonalize_mo and (not mix_mo):
            raise Warning("orthogonalize_mo=True has no effect as mix_mo=False")

        if orthogonalize_mo:
            raise Warning("Option orthogonalize_mo will be dprecated in 0.5.0")

        # # determine which orbs to include in the transformation
        self.include_all_mo = include_all_mo
        self.nmo_opt = self.mol.basis.nmo if include_all_mo else self.highest_occ_mo

        self.mo = MolecularOrbitals(
            self.mol,
            include_all_mo,
            self.highest_occ_mo,
            mix_mo,
            orthogonalize_mo,
            self.cuda,
        )

        if self.cuda:
            self.mo.to(self.device)

    def init_config(self, configs: str) -> None:
        """Initialize the electronic configurations desired in the wave function."""

        # define the SD we want
        self.orb_confs = OrbitalConfigurations(self.mol)
        if isinstance(configs, str):
            self.configs_method = configs
        elif isinstance(configs, tuple):
            self.configs_method = "explicit"
        self.configs = self.orb_confs.get_configs(configs)
        self.nci = len(self.configs[0])
        self.highest_occ_mo = max(self.configs[0].max(), self.configs[1].max()) + 1

    def init_slater_det_calculator(self) -> None:
        """Initialize the calculator of the slater dets"""

        #  define the SD pooling layer
        self.pool = SlaterPooling(
            self.configs_method, self.configs, self.mol, self.cuda
        )

    def init_fc_layer(self) -> None:
        """Init the fc layer"""

        # init the layer
        self.fc = nn.Linear(self.nci, 1, bias=False)

        # set all weight to 0 except the groud state
        self.fc.weight.data.fill_(0.0)
        self.fc.weight.data[0][0] = 1.0

        # port to card
        if self.cuda:
            self.fc = self.fc.to(self.device)

    def init_jastrow(self, jastrow: Union[str, nn.Module, None]) -> None:
        """Init the jastrow factor calculator"""

        # if the jastrow is explicitly None we disable the factor
        if jastrow is None:
            self.jastrow = jastrow
            self.use_jastrow = False

        # otherwise we use the jastrow provided by the user
        else:
            self.use_jastrow = True

            # create a simple Pade Jastrow factor as default
            if jastrow == "default":
                self.jastrow = JastrowFactorElectronElectron(
                    self.mol, PadeJastrowKernel, cuda=self.cuda
                )

            elif isinstance(jastrow, list):
                self.jastrow = CombineJastrow(jastrow)

            elif isinstance(jastrow, nn.Module):
                self.jastrow = jastrow

            else:
                raise TypeError("Jastrow factor not supported.")

            self.jastrow_type = self.jastrow.__repr__()
            if self.cuda:
                self.jastrow = self.jastrow.to(self.device)

    def set_combined_jastrow(self, jastrow: nn.Module):
        """Initialize the jastrow factor as a sum of jastrows"""
        self.jastrow = CombineJastrow(jastrow)

    def init_kinetic(
        self, kinetic: str, backflow: Union[BackFlowTransformation, None]
    ) -> None:
        """ "Init the calculator of the kinetic energies"""

        self.kinetic_method = kinetic
        if kinetic == "jacobi":
            if backflow is None:
                self.gradients = self.gradients_jacobi
                self.kinetic_energy = self.kinetic_energy_jacobi

            else:
                self.gradients = self.gradients_jacobi_backflow
                self.kinetic_energy = self.kinetic_energy_jacobi_backflow

    def forward(
        self, x: torch.Tensor, ao: Optional[Union[torch.Tensor, None]] = None
    ) -> torch.Tensor:
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

        # compute the jastrow from the pos
        if self.use_jastrow:
            J = self.jastrow(x)

        # atomic orbital
        if ao is None:
            x = self.ao(x)
        else:
            x = ao

        # molecular orbitals
        x = self.mo(x)

        # pool the mos
        x = self.pool(x)

        # compute the CI and return
        if self.use_jastrow:
            return J * self.fc(x)

        # if we do not have a Jastrow
        return self.fc(x)

    def ao2mo(self, ao: torch.Tensor) -> torch.Tensor:
        """transforms AO values in to MO values."""
        return self.mo(ao)

    def pos2mo(
        self,
        x: torch.Tensor,
        derivative: Optional[int] = 0,
        sum_grad: Optional[bool] = True,
    ) -> torch.Tensor:
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

    def kinetic_energy_jacobi(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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

    def gradients_jacobi(
        self,
        x: torch.Tensor,
        sum_grad: Optional[bool] = False,
        pdf: Optional[bool] = False,
    ) -> torch.Tensor:
        """Compute the gradients of the wave function (or density) using the Jacobi Formula
        C. Filippi, Simple Formalism for Efficient Derivatives.

        .. math::
             \\frac{K(R)}{\Psi(R)} = Tr(A^{-1} B_{grad})

        The gradients of the wave function

        .. math::
            \\Psi(R) = J(R) \\sum_n c_n D^{u}_n D^{d}_n = J(R) \\Sigma

        are computed following

        .. math::
            \\nabla \\Psi(R) = \\left( \\nabla J(R) \\right) \\Sigma + J(R) \\left(\\nabla \\Sigma \\right)

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
            out = jast * out

            # add the product (\nabla J(R)) \Sigma
            out = out + (grad_jast * self.fc(dets).unsqueeze(-1)).reshape(nbatch, -1)

        # compute the gradient of the pdf (i.e. the square of the wave function)
        # \nabla f^2 = 2 (\nabla f) f
        if pdf:
            out = 2 * out * self.fc(dets)
            if self.use_jastrow:
                out = out * jast

        return out

    def get_kinetic_operator(
        self,
        x: torch.Tensor,
        ao: torch.Tensor,
        dao: torch.Tensor,
        d2ao: torch.Tensor,
        mo: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Bkin matrix

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)
            mo (torch.tensor, optional): precomputed values of the MOs

        Returns:
            torch.tensor: matrix of the kinetic operator
        """

        bkin = self.ao2mo(d2ao)

        if self.use_jastrow:
            jast, djast, d2jast = self.jastrow(x, derivative=[0, 1, 2], sum_grad=False)

            djast = djast.transpose(1, 2) / jast.unsqueeze(-1)
            d2jast = d2jast / jast

            dmo = self.ao2mo(dao.transpose(2, 3)).transpose(2, 3)

            djast_dmo = (djast.unsqueeze(2) * dmo).sum(-1)
            d2jast_mo = d2jast.unsqueeze(-1) * mo

            bkin = bkin + 2 * djast_dmo + d2jast_mo

        return -0.5 * bkin

    def kinetic_energy_jacobi_backflow(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the value of the kinetic enery using the Jacobi Formula.


        .. math::
             \\frac{\\Delta (J(R) \\Psi(R))}{ J(R) \\Psi(R)} = \\frac{\\Delta J(R)}{J(R}
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
        ao, dao, d2ao = self.ao(x, derivative=[0, 1, 2], sum_grad=False)

        # get the mo values
        mo = self.ao2mo(ao)
        dmo = self.ao2mo(dao)
        d2mo = self.ao2mo(d2ao)

        # precompute the inverse of the MOs
        inv_mo = self.pool.compute_inverse_occupied_mo_matrix(mo)

        # compute the value of the slater det
        slater_dets = self.pool(mo)
        sum_slater_dets = self.fc(slater_dets)

        # compute ( tr(A_u^-1\Delta A_u) + tr(A_d^-1\Delta A_d) )
        hess = self.pool.operator(mo, d2mo, inv_mo=inv_mo)

        # compute (tr(A_u^-1\nabla A_u) and tr(A_d^-1\nabla A_d))
        grad = self.pool.operator(mo, dmo, op=None, inv_mo=inv_mo)

        # compute (tr((A_u^-1\nabla A_u)^2) + tr((A_d^-1\nabla A_d))^2)
        grad2 = self.pool.operator(mo, dmo, op_squared=True, inv_mo=inv_mo)

        # assemble the total second derivative term
        hess = (
            hess.sum(0)
            + operator.add(*[(g**2).sum(0) for g in grad])
            - grad2.sum(0)
            + 2 * operator.mul(*grad).sum(0)
        )

        hess = self.fc(hess * slater_dets) / sum_slater_dets

        if self.use_jastrow is False:
            return -0.5 * hess

        # compute the Jastrow terms
        jast, djast, d2jast = self.jastrow(x, derivative=[0, 1, 2], sum_grad=False)

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

        grad_val = self.fc(operator.add(*grad) * slater_dets) / sum_slater_dets

        # [Nelec*Ndim] x Nbatch
        grad_val = grad_val.squeeze()

        # assemble the derivaite terms
        out = d2jast.sum(-1) + 2 * (grad_val * djast).sum(0) + hess.squeeze(-1)
        return -0.5 * out.unsqueeze(-1)

    def gradients_jacobi_backflow(
        self,
        x: torch.Tensor,
        sum_grad: Optional[bool] = True,
        pdf: Optional[bool] = False,
    ):
        """Computes the gradients of the wf using Jacobi's Formula

        Args:
            x ([type]): [description]
        """
        raise NotImplementedError(
            "Gradient through Jacobi formula not implemented for backflow orbitals"
        )

    def log_data(self) -> None:
        """Print information abut the wave function."""
        log.info("")
        log.info(" Wave Function")
        log.info("  Backflow            : {0}", self.use_backflow)
        if self.use_backflow:
            log.info("  Backflow kernel     : {0}", self.backflow_type)
        log.info("  Jastrow factor      : {0}", self.use_jastrow)
        if self.use_jastrow:
            log.info("  Jastrow kernel      : {0}", self.jastrow_type)
        log.info("  Highest MO included : {0}", self.nmo_opt)
        log.info("  Orthogonalize MOs   : {0}", self.mo.orthogonalize_mo)
        log.info("  Configurations      : {0}", self.configs_method)
        log.info("  Number of confs     : {0}", self.nci)

        log.debug("  Configurations      : ")
        for ic in range(self.nci):
            cstr = " " + " ".join([str(i) for i in self.configs[0][ic].tolist()])
            cstr += " | " + " ".join([str(i) for i in self.configs[1][ic].tolist()])
            log.debug(cstr)

        log.info("  Kinetic energy      : {0}", self.kinetic_method)
        log.info("  Number var  param   : {0}", self.get_number_parameters())
        log.info("  Cuda support        : {0}", self.cuda)
        if self.cuda:
            log.info("  GPU                 : {0}", torch.cuda.get_device_name(0))

    def update_mo_coeffs(self):
        """Update the Mo coefficient during a GO run."""
        self.mol.atom_coords = self.ao.atom_coords.detach().numpy().tolist()
        self.mo.weight = self.get_mo_coeffs()

    def geometry(
        self, pos: torch.Tensor, convert_to_angs: Optional[bool] = False
    ) -> List:
        """Returns the gemoetry of the system in xyz format

        Args:
            pos (torch.tensor): sampling points (Nbatch, 3*Nelec)

        Returns:
            list: list where each element is one line of the xyz file
        """
        d = []
        convert = 1
        if convert_to_angs:
            convert = BOHR2ANGS
        for iat in range(self.natom):
            xyz = self.ao.atom_coords[iat, :].cpu().detach().numpy() * convert
            d.append(xyz.tolist())
        return d

    def gto2sto(self, plot: Optional[bool] = False) -> Self:
        """Fits the AO GTO to AO STO.
        The SZ sto that have only one basis function per ao
        """

        assert self.ao.radial_type.startswith("gto")
        assert self.ao.harmonics_type == "cart"

        log.info("  Fit GTOs to STOs  : ")

        def sto(x, norm, alpha):
            """Fitting function."""
            return norm * np.exp(-alpha * np.abs(x))

        # shortcut for nao
        nao = self.mol.basis.nao

        # create a new mol and a new basis
        new_mol = deepcopy(self.mol)
        basis = deepcopy(self.mol.basis)

        # change basis to sto
        basis.radial_type = "sto_pure"
        basis.nshells = self.ao.nao_per_atom.detach().cpu().numpy()

        # reset basis data
        basis.index_ctr = np.arange(nao)
        basis.bas_coeffs = np.ones(nao)
        basis.bas_exp = np.zeros(nao)
        basis.bas_norm = np.zeros(nao)
        basis.bas_kr = np.zeros(nao)
        basis.bas_kx = np.zeros(nao)
        basis.bas_ky = np.zeros(nao)
        basis.bas_kz = np.zeros(nao)

        # 2D fit space
        x = torch.linspace(-5, 5, 501)

        # compute the values of the current AOs using GTO BAS
        pos = x.reshape(-1, 1).repeat(1, self.ao.nbas).to(self.device)
        gto = self.ao.norm_cst * torch.exp(-self.ao.bas_exp * pos**2)
        gto = gto.unsqueeze(1).repeat(1, self.nelec, 1)
        ao = self.ao._contract(gto)[:, 0, :].detach().cpu().numpy()

        # loop over AOs
        for iorb in range(self.ao.norb):
            # fit AO with STO
            xdata = x.numpy()
            ydata = ao[:, iorb]
            popt, _ = curve_fit(sto, xdata, ydata)

            # store new exp/norm
            basis.bas_norm[iorb] = popt[0]
            basis.bas_exp[iorb] = popt[1]

            # determine k values
            basis.bas_kx[iorb] = (
                self.ao.harmonics.bas_kx[self.ao.index_ctr == iorb].unique().item()
            )
            basis.bas_ky[iorb] = (
                self.ao.harmonics.bas_ky[self.ao.index_ctr == iorb].unique().item()
            )
            basis.bas_kz[iorb] = (
                self.ao.harmonics.bas_kz[self.ao.index_ctr == iorb].unique().item()
            )

            # plot if necessary
            if plot:
                plt.plot(xdata, ydata)
                plt.plot(xdata, sto(xdata, *popt))
                plt.show()

        # update basis in new mole
        new_mol.basis = basis

        # returns new orbital instance
        return self.__class__(
            new_mol,
            self.jastrow,
            backflow=self.ao.backflow_trans,
            configs=self.configs_method,
            kinetic=self.kinetic_method,
            cuda=self.cuda,
            include_all_mo=self.include_all_mo,
        )
