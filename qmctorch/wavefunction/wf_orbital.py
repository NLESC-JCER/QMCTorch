import torch
from torch import nn
from copy import deepcopy

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

from .atomic_orbitals import AtomicOrbitals
from .slater_pooling import SlaterPooling
from .kinetic_pooling import KineticPooling
from .orbital_configurations import OrbitalConfigurations
from .wf_base import WaveFunction
from .pade_jastrow import PadeJastrow
from .scaled_pade_jastrow import ScaledPadeJastrow
from .pade_jastrow_polynomial import PadeJastrowPolynomial

from ..utils import register_extra_attributes

from .. import log


class Orbital(WaveFunction):

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

        super(Orbital, self).__init__(mol.nelec, 3, kinetic, cuda)

        # check for cuda
        if not torch.cuda.is_available and self.cuda:
            raise ValueError('Cuda not available, use cuda=False')

        # number of atoms
        self.mol = mol
        self.atoms = mol.atoms
        self.natom = mol.natom

        # define the SD we want
        self.orb_confs = OrbitalConfigurations(mol)
        self.configs_method = configs
        self.configs = self.orb_confs.get_configs(configs)
        self.nci = len(self.configs[0])
        self.highest_occ_mo = torch.stack(self.configs).max()+1

        # define the atomic orbital layer
        self.ao = AtomicOrbitals(mol, cuda)

        # define the mo layer
        self.include_all_mo = include_all_mo
        self.nmo_opt = mol.basis.nmo if include_all_mo else self.highest_occ_mo
        self.mo_scf = nn.Linear(
            mol.basis.nao, self.nmo_opt, bias=False)
        self.mo_scf.weight = self.get_mo_coeffs()
        self.mo_scf.weight.requires_grad = False
        if self.cuda:
            self.mo_scf.to(self.device)

        # define the mo mixing layer
        self.mo = nn.Linear(mol.basis.nmo, self.nmo_opt, bias=False)
        self.mo.weight = nn.Parameter(
            torch.eye(mol.basis.nmo, self.nmo_opt))
        if self.cuda:
            self.mo.to(self.device)

        # jastrow
        self.set_jastrow(use_jastrow, jastrow_type)

        #  define the SD pooling layer
        self.pool = SlaterPooling(self.configs_method,
                                  self.configs, mol, cuda)

        # pooling operation to directly compute
        # the kinetic energies via Jacobi formula
        self.kinpool = KineticPooling(
            self.configs, mol, cuda)

        # define the linear layer
        self.fc = nn.Linear(self.nci, 1, bias=False)
        self.fc.weight.data.fill_(0.)
        self.fc.weight.data[0][0] = 1.

        if self.cuda:
            self.fc = self.fc.to(self.device)

        self.kinetic_method = kinetic
        if kinetic == 'jacobi':
            self.local_energy = self.local_energy_jacobi

        if self.cuda:
            self.device = torch.device('cuda')
            self.to(self.device)

        # register the callable for hdf5 dump
        register_extra_attributes(self,
                                  ['ao', 'mo_scf',
                                   'mo', 'jastrow',
                                   'pool', 'kinpool', 'fc'])

        self.log_data()

    def log_data(self):
        log.info('')
        log.info(' Wave Function')
        log.info('  Jastrow factor      : {0}', self.use_jastrow)
        if self.use_jastrow:
            log.info('  Jastrow type        : {0}', self.jastrow_type)
        log.info('  Highest MO included : {0}', self.nmo_opt)
        log.info('  Configurations      : {0}', self.configs_method)
        log.info('  Number of confs     : {0}', self.nci)

        log.debug('  Configurations      : ')
        for ic in range(self.nci):
            cstr = ' ' + ' '.join([str(i)
                                   for i in self.configs[0][ic].tolist()])
            cstr += ' | ' + ' '.join([str(i)
                                      for i in self.configs[1][ic].tolist()])
            log.debug(cstr)

        log.info('  Kinetic energy      : {0}', self.kinetic_method)
        log.info(
            '  Number var  param   : {0}', self.get_number_parameters())
        log.info('  Cuda support        : {0}', self.cuda)
        if self.cuda:
            log.info(
                '  GPU                 : {0}', torch.cuda.get_device_name(0))

    def get_mo_coeffs(self):
        mo_coeff = torch.tensor(self.mol.basis.mos).type(
            torch.get_default_dtype())
        if not self.include_all_mo:
            mo_coeff = mo_coeff[:, :self.highest_occ_mo]
        return nn.Parameter(mo_coeff.transpose(0, 1).contiguous())

    def update_mo_coeffs(self):
        self.mol.atom_coords = self.ao.atom_coords.detach().numpy().tolist()
        self.mo.weight = self.get_mo_coeffs()

    def set_jastrow(self, use_jastrow, jastrow_type):
        """Set the jastrow calculator

        Args:
            jastrow_type (str): name of the jastrow
        """
        self.use_jastrow = use_jastrow
        self.jastrow_type = jastrow_type

        if jastrow_type == 'pade_jastrow':
            self.jastrow = PadeJastrow(self.mol.nup, self.mol.ndown,
                                       w=1., cuda=self.cuda)

        elif jastrow_type.startswith('pade_jastrow('):
            order = int(jastrow_type.split('(')[1][0])
            self.jastrow = PadeJastrowPolynomial(
                self.mol.nup, self.mol.ndown, order, cuda=self.cuda)

        elif jastrow_type == 'scaled_pade_jastrow':
            self.jastrow = ScaledPadeJastrow(self.mol.nup, self.mol.ndown,
                                             w=1., kappa=0.6, cuda=self.cuda)

        else:
            valid_names = ['pade_jastrow',
                           'pade_jastrow_(n)',
                           'scaled_pade_jastrow']
            log.info(
                '   Error : Jastrow form not recognized. Options are :')
            for n in valid_names:
                log.info('         : {0}', n)
            raise ValueError('Jastrow type not supported')

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

    def _get_mo_vals(self, x, derivative=0):
        """Get the values of MOs

        Arguments:
            x {torch.tensor} -- positions of the electrons [nbatch, nelec*ndim]

        Keyword Arguments:
            derivative {int} -- order of the derivative (default: {0})

        Returns:
            torch.tensor -- MO matrix [nbatch, nelec, nmo]
        """
        return self.mo(self.mo_scf(self.ao(x, derivative=derivative)))

    def local_energy_jacobi(self, pos):
        """Computes the local energy using the Jacobi formula

        .. math::
            E = K(R) + V_{ee}(R) + V_{en}(R) + V_{nn}

        Args:
            pos (torch.tensor): sampling points (Nbatch, 3*Nelec)

        Returns:
            [torch.tensor]: values of the local enrgies at each sampling points

        Examples::
            >>> mol = Molecule('h2.xyz', calculator='adf', basis = 'dzp')
            >>> wf = Orbital(mol, configs='cas(2,2)')
            >>> pos = torch.rand(500,6)
            >>> vals = wf.local_energy_jacobi(pos)

        """

        ke = self.kinetic_energy_jacobi(pos)

        return ke \
            + self.nuclear_potential(pos) \
            + self.electronic_potential(pos) \
            + self.nuclear_repulsion()

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
            kin = self.pool.kinetic(mo, bkin)
            psi = self.pool(mo)
            out = self.fc(kin * psi) / self.fc(psi)
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
                                               jacobian=False)

            djast = djast.transpose(1, 2) / jast.unsqueeze(-1)
            d2jast = d2jast / jast

            dmo = self.ao2mo(dao.transpose(2, 3)).transpose(2, 3)

            djast_dmo = (djast.unsqueeze(2) * dmo).sum(-1)
            d2jast_mo = d2jast.unsqueeze(-1) * mo

            bkin = bkin + 2 * djast_dmo + d2jast_mo

        return bkin

    def geometry(self, pos):
        """Returns the gemoetry of the system in xyz format

        Args:
            pos (torch.tensor): sampling points (Nbatch, 3*Nelec)

        Returns:
            list: list where each element is one line of the xyz file
        """
        d = []
        for iat in range(self.natom):
            at = self.atoms[iat]
            xyz = self.ao.atom_coords[iat,
                                      :].detach().numpy().tolist()
            d.append((at, xyz))
        return d

    def gto2sto(self, plot=False):
        """Fits the AO GTO to AO STO.
            The sto have only one basis function per ao
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
