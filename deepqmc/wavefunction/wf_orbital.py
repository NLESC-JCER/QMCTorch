import torch
from torch import nn
import numpy as np

from deepqmc.wavefunction.wf_base import WaveFunction
from deepqmc.wavefunction.atomic_orbitals import AtomicOrbitals
from deepqmc.wavefunction.slater_pooling import SlaterPooling
from deepqmc.wavefunction.kinetic_pooling import KineticPooling
from deepqmc.wavefunction.jastrow import TwoBodyJastrowFactor


class Orbital(WaveFunction):

    def __init__(self, mol, configs='ground_state', scf='pyscf',
                 kinetic='jacobi', use_jastrow=True, cuda=False):

        super(Orbital, self).__init__(mol.nelec, 3, kinetic, cuda)

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
        self.mo = nn.Linear(mol.norb, mol.norb, bias=False)
        self.mo.weight = self.get_mo_coeffs()
        if self.cuda:
            self.mo.to(self.device)

        # jastrow
        self.use_jastrow = use_jastrow
        if self.use_jastrow:
            self.jastrow = TwoBodyJastrowFactor(mol.nup, mol.ndown,
                                                w=1., cuda=cuda)

        # define the SD we want
        self.configs = self.get_configs(configs, mol)
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

    def get_mo_coeffs(self):
        mo_coeff = torch.tensor(
            self.mol.get_mo_coeffs(code=self.scf_code)).type(torch.get_default_dtype())
        return nn.Parameter(mo_coeff.transpose(0, 1))

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

        x = self.ao(x)
        x = self.mo(x)
        x = self.pool(x)

        if self.use_jastrow:
            return J*self.fc(x)

        return self.fc(x)

    def local_energy_jacobi(self, pos):
        ''' local energy of the sampling points.'''

        ke = self.kinetic_energy_jacobi(pos, return_local_energy=True)

        return ke \
            + self.nuclear_potential(pos) \
            + self.electronic_potential(pos) \
            + self.nuclear_repulsion()

    def kinetic_energy_jacobi(self, x, return_local_energy=False, **kwargs):
        '''Compute the value of the kinetic enery using
        the Jacobi formula for derivative of determinant.

        Args:
            x: position of the electrons

        Returns: values of \Delta \Psi
        '''

        MO = self.mo(self.ao(x))
        d2MO = self.mo(self.ao(x, derivative=2))
        dJdMO, d2JMO = None, None

        if self.use_jastrow:

            J = self.jastrow(x)
            dJ = self.jastrow(x, derivative=1) / J
            d2J = self.jastrow(x, derivative=2) / J

            dJdMO = dJ.unsqueeze(-1) * self.mo(self.ao(x, derivative=1))
            d2JMO = d2J.unsqueeze(-1) * MO

        return self.fc(self.kinpool(MO, d2MO, dJdMO, d2JMO,
                                    return_local_energy=return_local_energy))

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
                r = torch.sqrt(((pelec-patom)**2).sum(1)) + 1E-6
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
                r = torch.sqrt(((epos1-epos2)**2).sum(1)) + 1E-6
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

    def get_configs(self, configs, mol):
        """Get the configuratio in the CI expansion

        Args:
            configs (str): name of the configs we want
            mol (mol object): molecule object

        Returns:
            tuple(torch.LongTensor,torch.LongTensor): the spin up/spin down
            electronic confs
        """
        if isinstance(configs, torch.Tensor):
            return configs

        elif configs == 'ground_state':
            return self._get_ground_state_config(mol)

        elif configs.startswith('singlet'):
            nocc, nvirt = eval(configs.lstrip("singlet"))
            return self._get_singlet_state_config(mol, nocc, nvirt)

        else:
            raise ValueError(configs, " not recognized as valid configuration")

    @staticmethod
    def _get_ground_state_config(mol):
        """Return only the ground state configuration

        Args:
            mol (mol): mol object

        Returns:
            tuple(torch.LongTensor,torch.LongTensor): the spin up/spin down
            electronic confs
        """
        conf = (torch.LongTensor([np.array(range(mol.nup))]),
                torch.LongTensor([np.array(range(mol.ndown))]))
        return conf

    def _get_singlet_state_config(self, mol, nocc, nvirt):
        """Get the confs of the singlet conformations

        Args:
            mol (mol): mol object
            nocc (int): number of occupied orbitals in the active space
            nvirt (int): number of virtual orbitals in the active space
        """

        _gs = list(range(mol.nup))
        cup, cdown = [_gs], [_gs]

        for ivirt in range(mol.nup, mol.nup+nvirt, 1):
            for iocc in range(mol.nup-1, mol.nup-1-nocc, -1):

                _xt = self._create_excitation(_gs.copy(), iocc, ivirt)
                cup, cdown = self._append_excitations(cup, cdown, _xt, _gs)
                cup, cdown = self._append_excitations(cup, cdown, _gs, _xt)

        return (torch.LongTensor(cup), torch.LongTensor(cdown))

    @staticmethod
    def _create_excitation(conf, iocc, ivirt):
        """promote an electron from iocc to ivirt

        Args:
            conf (list): index of the occupied orbitals
            iocc (int): index of the occupied orbital
            ivirt (int): index of the virtual orbital

        Returns:
            list: new configuration
        """
        conf.pop(iocc)
        conf += [ivirt]
        return conf

    @staticmethod
    def _append_excitations(cup, cdown, new_cup, new_cdown):
        """Append new excitations

        Args:
            cup (list): configurations of spin up
            cdown (list): configurations of spin down
            new_cup (list): new spin up confs
            new_cdown (list): new spin down confs
        """

        cup.append(new_cup)
        cdown.append(new_cdown)
        return cup, cdown


if __name__ == "__main__":

    from deepqmc.wavefunction.molecule import Molecule

    mol = Molecule(atom='Li 0 0 0; H 0 0 3.015',
                   basis_type='gto', basis='sto-3g')

    # define the wave function
    wf = Orbital(mol, kinetic='jacobi',
                 configs='singlet(1,1)',
                 use_jastrow=True, cuda=True)
