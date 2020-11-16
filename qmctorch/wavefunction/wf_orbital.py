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
from .jastrows.jastrow import set_jastrow


class Orbital(OrbitalBase):

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

        super(Orbital, self).__init__(mol, configs, kinetic,
                                      use_jastrow, jastrow_type, cuda, include_all_mo)

        self.jastrow = set_jastrow(
            jastrow_type, self.mol.nup, self.mol.ndown, self.cuda)

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

        if self.use_jastrow:
            J = self.jastrow(x)
            print(J.shape)
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

    def kinetic_energy_jacobi(self, x, kinpool=False, **kwargs):
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

        ao, dao, d2ao = self.ao(x, derivative=[0, 1, 2])
        mo = self.ao2mo(ao)
        bkin = self.get_kinetic_operator(x, ao, dao, d2ao, mo)

        if kinpool:
            kin, psi = self.kinpool(mo, bkin)
            return self.fc(kin) / self.fc(psi)

        else:
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

    def geometry(self, pos):
        """Returns the gemoetry of the system in xyz format

        Args:
            pos (torch.tensor): sampling points (Nbatch, 3*Nelec)

        Returns:
            list: list where each element is one line of the xyz file
        """
        d = []
        for iat in range(self.natom):

            xyz = self.ao.atom_coords[iat,
                                      :].cpu().detach().numpy().tolist()
            d.append(xyz)
        return d

    def gto2sto(self, plot=False):
        """Fits the AO GTO to AO STO.
            The SZ sto tht have only one basis function per ao
        """

        assert(self.ao.radial_type.startswith('gto'))
        assert(self.ao.harmonics_type == 'cart')

        log.info('  Fit GTOs to STOs  : ')

        def sto(x, norm, alpha):
            """Fitting function."""
            return norm * np.exp(-alpha * np.abs(x))

        # shortcut for nao
        nao = self.mol.basis.nao

        # create a new mol and a new basis
        new_mol = deepcopy(self.mol)
        basis = deepcopy(self.mol.basis)

        # change basis to sto
        basis.radial_type = 'sto_pure'
        basis.nshells = self.ao.nao_per_atom.numpy()

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
        gto = self.ao.norm_cst * torch.exp(-self.ao.bas_exp*pos**2)
        gto = gto.unsqueeze(1).repeat(1, self.nelec, 1)
        ao = self.ao._contract(gto)[
            :, 0, :].detach().cpu().numpy()

        # loop over AOs
        for iorb in range(self.ao.norb):

            # fit AO with STO
            xdata = x.numpy()
            ydata = ao[:, iorb]
            popt, pcov = curve_fit(sto, xdata, ydata)

            # store new exp/norm
            basis.bas_norm[iorb] = popt[0]
            basis.bas_exp[iorb] = popt[1]

            # determine k values
            basis.bas_kx[iorb] = self.ao.harmonics.bas_kx[self.ao.index_ctr == iorb].unique(
            ).item()
            basis.bas_ky[iorb] = self.ao.harmonics.bas_ky[self.ao.index_ctr == iorb].unique(
            ).item()
            basis.bas_kz[iorb] = self.ao.harmonics.bas_kz[self.ao.index_ctr == iorb].unique(
            ).item()

            # plot if necessary
            if plot:
                plt.plot(xdata, ydata)
                plt.plot(xdata, sto(xdata, *popt))
                plt.show()

        # update basis in new mole
        new_mol.basis = basis

        # returns new orbital instance
        return Orbital(new_mol, configs=self.configs_method,
                       kinetic=self.kinetic_method,
                       use_jastrow=self.use_jastrow,
                       jastrow_type=self.jastrow_type,
                       cuda=self.cuda,
                       include_all_mo=self.include_all_mo)
