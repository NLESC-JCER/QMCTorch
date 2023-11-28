from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from torch import nn

import torch
from .. import log
from ..utils import register_extra_attributes
from .orbitals.atomic_orbitals import AtomicOrbitals
from .pooling.orbital_configurations import OrbitalConfigurations
from .pooling.slater_pooling import SlaterPooling
from .wf_base import WaveFunction


class SlaterJastrowBase(WaveFunction):

    def __init__(self, mol,
                 configs='ground_state',
                 kinetic='jacobi',
                 cuda=False,
                 include_all_mo=True):
        """Implementation of the QMC Network.

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
            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
            include_all_mo (bool, optional): include either all molecular orbitals or only the ones that are
                                             popualted in the configs. Defaults to False
        """

        super(SlaterJastrowBase, self).__init__(
            mol.nelec, 3, kinetic, cuda)

        # check for cuda
        if not torch.cuda.is_available and self.cuda:
            raise ValueError('Cuda not available, use cuda=False')

        # check for conf/mo size
        if not include_all_mo and configs.startswith('cas('):
            raise ValueError(
                'CAS calculation only possible with include_all_mo=True')

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
        # self.mo = nn.Linear(mol.basis.nmo, self.nmo_opt, bias=False)
        self.mo = nn.Linear(self.nmo_opt, self.nmo_opt, bias=False)
        self.mo.weight = nn.Parameter(
            torch.eye(self.nmo_opt, self.nmo_opt))
        if self.cuda:
            self.mo.to(self.device)

        # jastrow
        self.jastrow_type = None
        self.use_jastrow = False

        #  define the SD pooling layer
        self.pool = SlaterPooling(self.configs_method,
                                  self.configs, mol, cuda)

        # define the linear layer
        self.fc = nn.Linear(self.nci, 1, bias=False)
        self.fc.weight.data.fill_(0.)
        self.fc.weight.data[0][0] = 1.

        if self.cuda:
            self.fc = self.fc.to(self.device)

        self.kinetic_method = kinetic
        if kinetic == 'jacobi':
            self.kinetic_energy = self.kinetic_energy_jacobi

        gradients = 'auto'
        self.gradients_method = gradients
        if gradients == 'jacobi':
            self.gradients = self.gradients_jacobi

        if self.cuda:
            self.device = torch.device('cuda')
            self.to(self.device)

        # register the callable for hdf5 dump
        register_extra_attributes(self,
                                  ['ao', 'mo_scf',
                                   'mo', 'jastrow',
                                   'pool', 'fc'])

    def log_data(self):
        """Print information abut the wave function."""
        log.info('')
        log.info(' Wave Function')
        log.info('  Jastrow factor      : {0}', self.use_jastrow)
        if self.use_jastrow:
            log.info(
                '  Jastrow kernel      : {0}', self.jastrow_type)
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
        mo_coeff = torch.as_tensor(self.mol.basis.mos).type(
            torch.get_default_dtype())
        if not self.include_all_mo:
            mo_coeff = mo_coeff[:, :self.highest_occ_mo]
        return nn.Parameter(mo_coeff.transpose(0, 1).contiguous())

    def update_mo_coeffs(self):
        self.mol.atom_coords = self.ao.atom_coords.detach().numpy().tolist()
        self.mo.weight = self.get_mo_coeffs()

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
            The SZ sto that have only one basis function per ao
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
        return self.__class__(new_mol, configs=self.configs_method,
                              kinetic=self.kinetic_method,
                              cuda=self.cuda,
                              include_all_mo=self.include_all_mo)

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

        raise NotImplementedError('Implement a forward method')

    def ao2mo(self, ao):
        """Get the values of the MO from the values of AO."""
        raise NotImplementedError('Implement a ao2mo method')

    def pos2mo(self, x, derivative=0):
        """Get the values of MOs from the positions

        Arguments:
            x {torch.tensor} -- positions of the electrons [nbatch, nelec*ndim]

        Keyword Arguments:
            derivative {int} -- order of the derivative (default: {0})

        Returns:
            torch.tensor -- MO matrix [nbatch, nelec, nmo]
        """
        raise NotImplementedError('Implement a get_mo_vals method')

    def kinetic_energy_jacobi(self, x,  **kwargs):
        """Compute the value of the kinetic enery using the Jacobi Formula.
        C. Filippi, Simple Formalism for Efficient Derivatives .

        .. math::
             \\frac{K(R)}{\\Psi(R)} = Tr(A^{-1} B_{kin})

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)

        Returns:
            torch.tensor: values of the kinetic energy at each sampling points
        """

        raise NotImplementedError(
            'Implement a kinetic_energy_jacobi method')

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

        raise NotImplementedError(
            'Implement a gradient_jacobi method')

    def get_gradient_operator(self, x, ao, grad_ao, mo):
        """Compute the gradient operator

        Args:
            x ([type]): [description]
            ao ([type]): [description]
            dao ([type]): [description]
        """

        raise NotImplementedError(
            'Implement a get_grad_operator method')

    def get_hessian_operator(self, x, ao, dao, d2ao,  mo):
        """Compute the Bkin matrix

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)
            mo (torch.tensor, optional): precomputed values of the MOs

        Returns:
            torch.tensor: matrix of the kinetic operator
        """

        raise NotImplementedError(
            'Implement a get_kinetic_operator method')
