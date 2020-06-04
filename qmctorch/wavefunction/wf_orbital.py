import torch
from torch import nn

from .atomic_orbitals import AtomicOrbitals
from .slater_pooling import SlaterPooling
from .kinetic_pooling import KineticPooling
from .orbital_configurations import OrbitalConfigurations
from .wf_base import WaveFunction
from .fast_jastrow import TwoBodyJastrowFactor
#from .jastrow import TwoBodyJastrowFactor

from ..utils import register_extra_attributes
from ..utils.interpolate import (get_reg_grid, get_log_grid,
                                 interpolator_reg_grid, interpolate_reg_grid,
                                 interpolator_irreg_grid, interpolate_irreg_grid)
from .. import log


class Orbital(WaveFunction):

    def __init__(self, mol, configs='ground_state',
                 kinetic='jacobi',
                 use_jastrow=True, cuda=False,
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
        if not torch.cuda.is_available and self.wf.cuda:
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
        self.use_jastrow = use_jastrow
        self.jastrow = TwoBodyJastrowFactor(mol.nup, mol.ndown,
                                            w=1., cuda=cuda)

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
        log.info('  Highest MO included : {0}', self.highest_occ_mo)
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

    def get_mo_coeffs(self):
        mo_coeff = torch.tensor(self.mol.basis.mos).type(
            torch.get_default_dtype())
        if not self.include_all_mo:
            mo_coeff = mo_coeff[:, :self.highest_occ_mo]
        return nn.Parameter(mo_coeff.transpose(0, 1).contiguous())

    def update_mo_coeffs(self):
        self.mol.atom_coords = self.ao.atom_coords.detach().numpy().tolist()
        self.mo.weight = self.get_mo_coeffs()

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

        mo = self._get_mo_vals(x)
        bkin = self.get_kinetic_operator(x, mo=mo)

        if kinpool:
            kin, psi = self.kinpool(mo, bkin)
            return self.fc(kin) / self.fc(psi)

        else:
            kin = self.pool.kinetic(mo, bkin)
            psi = self.pool(mo)
            out = self.fc(kin * psi) / self.fc(psi)
            return out

    def get_kinetic_operator(self, x, mo=None):
        """Compute the Bkin matrix

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)
            mo (torch.tensor, optional): precomputed values of the MOs

        Returns:
            torch.tensor: matrix of the kinetic operator
        """

        ao, dao, d2ao = self.ao(
            x, derivative=[0, 1, 2], jacobian=False)

        if mo is None:
            mo = self._get_mo_vals(x)

        bkin = self._get_mo_vals(x, derivative=2)
        djast_dmo, d2jast_mo = None, None

        if self.use_jastrow:

            jast = self.jastrow(x)
            djast = self.jastrow(x, derivative=1, jacobian=False)
            djast = djast.transpose(1, 2) / jast.unsqueeze(-1)

            dao = self.ao(x, derivative=1,
                          jacobian=False).transpose(2, 3)
            dmo = self.mo(self.mo_scf(dao)).transpose(2, 3)

            djast_dmo = (djast.unsqueeze(2) * dmo).sum(-1)
            d2jast = self.jastrow(x, derivative=2) / jast
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
