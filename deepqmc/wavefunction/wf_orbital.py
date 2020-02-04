import torch
from torch import nn
import numpy as np


from deepqmc.wavefunction.atomic_orbitals import AtomicOrbitals
from deepqmc.wavefunction.slater_pooling import SlaterPooling
from deepqmc.wavefunction.kinetic_pooling import KineticPooling
from deepqmc.wavefunction.orbital_configurations import OrbitalConfigurations
from deepqmc.wavefunction.wf_base import WaveFunction
from deepqmc.wavefunction.jastrow import TwoBodyJastrowFactor


class Orbital(WaveFunction):

    def __init__(self, mol, configs='ground_state', scf='pyscf',
                 kinetic='jacobi', use_jastrow=True, cuda=False):

        super(Orbital, self).__init__(mol.nelec, 3, kinetic, cuda)

        # check for cuda
        if not torch.cuda.is_available and self.wf.cuda:
            raise ValueError('Cuda not available, use cuda=False')

        # number of atoms
        self.mol = mol
        self.atoms = mol.atoms
        self.bonds = mol.bonds
        self.natom = mol.natom

        # scf code
        self.scf_code = scf

        # define the atomic orbital layer
        self.ao = AtomicOrbitals(mol, cuda)

        # define the mo layer
        self.mo_scf = nn.Linear(mol.norb, mol.norb, bias=False)
        self.mo_scf.weight = self.get_mo_coeffs()
        self.mo_scf.weight.requires_grad = False
        if self.cuda:
            self.mo_scf.to(self.device)

        # define the mo mixing layer
        self.mo = nn.Linear(mol.norb, mol.norb, bias=False)
        self.mo.weight = nn.Parameter(torch.eye(mol.norb))
        if self.cuda:
            self.mo.to(self.device)

        # jastrow
        self.use_jastrow = use_jastrow
        self.jastrow = TwoBodyJastrowFactor(mol.nup, mol.ndown,
                                            w=1., cuda=cuda)

        # define the SD we want
        self.orb_confs = OrbitalConfigurations(mol)
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
        self.fc.weight.data.fill_(1.)
        if self.nci > 1:
            self.fc.weight.data.fill_(0.)
            self.fc.weight.data[0][0] = 1.
        if self.cuda:
            self.fc = self.fc.to(self.device)
        self.fc.clip = False

        if kinetic == 'jacobi':
            self.local_energy = self.local_energy_jacobi

        if self.cuda:
            self.device = torch.device('cuda')
            self.to(self.device)

    def get_mo_coeffs(self):
        mo_coeff = torch.tensor(
            self.mol.get_mo_coeffs(code=self.scf_code)).type(
                torch.get_default_dtype())
        return nn.Parameter(mo_coeff.transpose(0, 1).contiguous())

    def update_mo_coeffs(self):
        self.mol.atom_coords = self.ao.atom_coords.detach().numpy().tolist()
        self.mo.weight = self.get_mo_coeffs()

    def forward(self, x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            x: position of the electrons

        Returns: values of psi
        '''

        if self.use_jastrow:
            J = self.jastrow(x)

        # atomic orbital
        x = self.ao(x)

        # molecular orbitals
        x = self.mo_scf(x)

        # mix the mos
        x = self.mo(x)

        # pool the mos
        x = self.pool(x)

        if self.use_jastrow:
            return J*self.fc(x)
        else:
            return self.fc(x)

    def _get_mo_vals(self, x, derivative=0):
        '''get the values of the MOs.'''
        return self.mo(self.mo_scf(self.ao(x, derivative=derivative)))

    def local_energy_jacobi(self, pos):
        ''' local energy of the sampling points.'''

        ke = self.kinetic_energy_jacobi(pos)

        return ke \
            + self.nuclear_potential(pos) \
            + self.electronic_potential(pos) \
            + self.nuclear_repulsion()

    def kinetic_energy_jacobi(self, x, **kwargs):
        '''Compute the value of the kinetic enery using
        the Jacobi formula for derivative of determinant.

        Args:
            x: position of the electrons

        Returns: values of \Delta \Psi
        '''

        mo = self._get_mo_vals(x)
        d2mo = self._get_mo_vals(x, derivative=2)
        djast_dmo, d2jast_mo = None, None

        if self.use_jastrow:

            jast = self.jastrow(x)
            djast = self.jastrow(x, derivative=1, jacobian=False)
            djast = djast.transpose(1, 2) / jast.unsqueeze(-1)

            dao = self.ao(x, derivative=1, jacobian=False).transpose(2, 3)
            dmo = self.mo(self.mo_scf(dao)).transpose(2, 3)
            djast_dmo = (djast.unsqueeze(2) * dmo).sum(-1)

            d2jast = self.jastrow(x, derivative=2) / jast
            d2jast_mo = d2jast.unsqueeze(-1) * mo

        kin, psi = self.kinpool(mo, d2mo, djast_dmo, d2jast_mo)

        return self.fc(kin)/self.fc(psi)

    def nuclear_potential(self, pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi

        TODO : vecorize that !!
        '''

        p = torch.zeros(pos.shape[0], device=self.device)
        for ielec in range(self.nelec):
            pelec = pos[:, (ielec*self.ndim):(ielec+1)*self.ndim]
            for iatom in range(self.natom):
                patom = self.ao.atom_coords[iatom, :]
                Z = self.ao.atomic_number[iatom]
                r = torch.sqrt(((pelec-patom)**2).sum(1))  # + 1E-12
                p += (-Z/r)
        return p.view(-1, 1)

    def electronic_potential(self, pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''

        pot = torch.zeros(pos.shape[0], device=self.device)

        for ielec1 in range(self.nelec-1):
            epos1 = pos[:, ielec1*self.ndim:(ielec1+1)*self.ndim]
            for ielec2 in range(ielec1+1, self.nelec):
                epos2 = pos[:, ielec2*self.ndim:(ielec2+1)*self.ndim]
                r = torch.sqrt(((epos1-epos2)**2).sum(1))  # + 1E-12
                pot += (1./r)
        return pot.view(-1, 1)

    def nuclear_repulsion(self):
        '''Compute the nuclear repulsion term
        Returns: values of Vnn
        '''

        vnn = 0.
        for at1 in range(self.natom-1):
            c0 = self.ao.atom_coords[at1, :]
            Z0 = self.ao.atomic_number[at1]
            for at2 in range(at1+1, self.natom):
                c1 = self.ao.atom_coords[at2, :]
                Z1 = self.ao.atomic_number[at2]
                rnn = torch.sqrt(((c0-c1)**2).sum())
                vnn += Z0*Z1/rnn
        return vnn

    def atomic_distances(self, pos):
        '''Return atomic distances.'''
        d = []
        for iat1 in range(self.natom-1):
            at1 = self.atoms[iat1]
            c1 = self.ao.atom_coords[iat1, :].detach().numpy()
            for iat2 in range(iat1+1, self.natom):
                at2 = self.atoms[iat2]
                c2 = self.ao.atom_coords[iat2, :].detach().numpy()
                d.append((at1, at2, np.sum(np.sqrt(((c1-c2)**2)))))
        return d

    def geometry(self, pos):
        '''Return geometries.'''
        d = []
        for iat in range(self.natom):
            at = self.atoms[iat]
            xyz = self.ao.atom_coords[iat, :].detach().numpy().tolist()
            d.append((at, xyz))
        return d


if __name__ == "__main__":

    from deepqmc.wavefunction.molecule import Molecule

    mol = Molecule(atom='Li 0 0 0; H 0 0 3.015',
                   basis_type='sto', basis='sz')

    # mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
    #                     basis_type='sto', basis='sz',
    #                     unit='bohr')

    # define the wave function
    wf_jacobi = Orbital(mol, kinetic='jacobi',
                        configs='cas(2,2)',
                        use_jastrow=True,
                        cuda=False)

    wf_auto = Orbital(mol, kinetic='auto',
                      configs='cas(2,2)',
                      use_jastrow=True,
                      cuda=False)

    pos = torch.rand(20, wf_auto.ao.nelec*3)
    pos.requires_grad = True

    ej = wf_jacobi.energy(pos)
    ej.backward()

    ea = wf_auto.energy(pos)
    ea.backward()

    for p1,p2 in zip(wf_auto.parameters(),wf_jacobi.parameters()):
        if p1.requires_grad:
            print('')
            print(p1.grad)
            print(p2.grad)

    if torch.cuda.is_available():
        pos_gpu = pos.to('cuda')
        wf_gpu = Orbital(mol, kinetic='jacobi',
                         configs='singlet(1,1)',
                         use_jastrow=True, cuda=True)
