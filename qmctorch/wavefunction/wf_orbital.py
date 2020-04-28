import torch
from torch import nn

from .atomic_orbitals import AtomicOrbitals
from .slater_pooling import SlaterPooling
from .kinetic_pooling import KineticPooling
from .orbital_configurations import OrbitalConfigurations
from .wf_base import WaveFunction
from .jastrow import TwoBodyJastrowFactor

from ..utils import register_extra_attributes


class Orbital(WaveFunction):

    def __init__(self, mol, configs='ground_state',
                 kinetic='jacobi', use_jastrow=True, cuda=False):
        """Implementation of the QMC Network. 

        Args:
            mol (qmc.wavefunction.Molecule): a molecule object
            configs (str, optional): defines the CI configurations to be used. Defaults to 'ground_state'.
            kinetic (str, optional): method to compute the kinetic energy. Defaults to 'jacobi'.
            use_jastrow (bool, optional): turn jastrow factor ON/OFF. Defaults to True.
            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.

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

        # define the atomic orbital layer
        self.ao = AtomicOrbitals(mol, cuda)

        # define the mo layer
        self.mo_scf = nn.Linear(
            mol.basis.nao, mol.basis.nmo, bias=False)
        self.mo_scf.weight = self.get_mo_coeffs()
        self.mo_scf.weight.requires_grad = False
        if self.cuda:
            self.mo_scf.to(self.device)

        # define the mo mixing layer
        self.mo = nn.Linear(mol.basis.nmo, mol.basis.nmo, bias=False)
        self.mo.weight = nn.Parameter(torch.eye(mol.basis.nmo))
        if self.cuda:
            self.mo.to(self.device)

        # jastrow
        self.use_jastrow = use_jastrow
        self.jastrow = TwoBodyJastrowFactor(mol.nup, mol.ndown,
                                            w=1., cuda=cuda)

        # define the SD we want
        self.orb_confs = OrbitalConfigurations(mol)
        self.configs_method = configs
        self.configs = self.orb_confs.get_configs(configs)
        self.nci = len(self.configs[0])

        #  define the SD pooling layer
        self.pool = SlaterPooling(
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

    def get_mo_coeffs(self):
        mo_coeff = torch.tensor(self.mol.basis.mos).type(
            torch.get_default_dtype())
        return nn.Parameter(mo_coeff.transpose(0, 1).contiguous())

    def update_mo_coeffs(self):
        self.mol.atom_coords = self.ao.atom_coords.detach().numpy().tolist()
        self.mo.weight = self.get_mo_coeffs()

    def forward(self, x, ao=None):
        """computes the value of the wave function for the sampling points

        .. math::
            \Psi(R) = \sum_{n} c_n D^{u}_n(r^u) \\times D^{d}_n(r^d)

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

    def kinetic_energy_jacobi(self, x, **kwargs):
        """Compute the value of the kinetic enery using the Jacobi Formula.
        C. Filippi, Simple Formalism for Efficient Derivatives .

        .. math::
             \\frac{K(R)}{\Psi(R)} = Tr(A^{-1} B_{kin})

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)

        Returns:
            torch.tensor: values of the kinetic energy at each sampling points
        """

        mo = self._get_mo_vals(x)
        d2mo = self._get_mo_vals(x, derivative=2)
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

        kin, psi = self.kinpool(mo, d2mo, djast_dmo, d2jast_mo)

        return self.fc(kin) / self.fc(psi)

    def nuclear_potential(self, pos):
        """Computes the electron-nuclear term

        .. math:
            V_{en} = - \sum_e \sum_n \\frac{Z_n}{r_{en}}

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)

        Returns:
            torch.tensor: values of the electon-nuclear energy at each sampling points
        """

        p = torch.zeros(pos.shape[0], device=self.device)
        for ielec in range(self.nelec):
            istart = ielec * self.ndim
            iend = (ielec + 1) * self.ndim
            pelec = pos[:, istart:iend]
            for iatom in range(self.natom):
                patom = self.ao.atom_coords[iatom, :]
                Z = self.ao.atomic_number[iatom]
                r = torch.sqrt(((pelec - patom)**2).sum(1))  # + 1E-12
                p += -Z / r
        return p.view(-1, 1)

    def electronic_potential(self, pos):
        """Computes the electron-electron term

        .. math:
            V_{ee} = \sum_{e_1} \sum_{e_2} \\frac{1}{r_{e_1e_2}}

        Args:
            x (torch.tensor): sampling points (Nbatch, 3*Nelec)

        Returns:
            torch.tensor: values of the electon-electron energy at each sampling points
        """

        pot = torch.zeros(pos.shape[0], device=self.device)

        for ielec1 in range(self.nelec - 1):
            epos1 = pos[:, ielec1 *
                        self.ndim:(ielec1 + 1) * self.ndim]
            for ielec2 in range(ielec1 + 1, self.nelec):
                epos2 = pos[:, ielec2 *
                            self.ndim:(ielec2 + 1) * self.ndim]
                r = torch.sqrt(((epos1 - epos2)**2).sum(1))  # + 1E-12
                pot += (1. / r)
        return pot.view(-1, 1)

    def nuclear_repulsion(self):
        """Computes the nuclear-nuclear repulsion term

        .. math:
            V_{nn} = \sum_{n_1} \sum_{n_2} \\frac{Z_1Z_2}{r_{n_1n_2}}

        Returns:
            torch.tensor: values of the nuclear-nuclear energy at each sampling points
        """

        vnn = 0.
        for at1 in range(self.natom - 1):
            c0 = self.ao.atom_coords[at1, :]
            Z0 = self.ao.atomic_number[at1]
            for at2 in range(at1 + 1, self.natom):
                c1 = self.ao.atom_coords[at2, :]
                Z1 = self.ao.atomic_number[at2]
                rnn = torch.sqrt(((c0 - c1)**2).sum())
                vnn += Z0 * Z1 / rnn
        return vnn

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
