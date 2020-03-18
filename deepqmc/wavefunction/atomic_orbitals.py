import torch
from torch import nn
import numpy as np

from deepqmc.wavefunction.spherical_harmonics import SphericalHarmonics
from deepqmc.wavefunction.grad_spherical_harmonics import GradSphericalHarmonics

from time import time


class AtomicOrbitals(nn.Module):

    def __init__(self, mol, cuda=False):
        """Atomic Orbital Layer

        Arguments:
            mol {Molecule} -- Molecule instance

        Keyword Arguments:
            cuda {bool} -- use cuda (default: {False})
        """

        super(AtomicOrbitals, self).__init__()

        # wavefunction data
        self.nelec = mol.nelec
        self.norb = mol.norb
        self.ndim = 3

        # make the atomic position optmizable
        self.atom_coords = nn.Parameter(torch.tensor(mol.atom_coords))
        self.atom_coords.requires_grad = True
        self.natoms = len(self.atom_coords)
        self.atomic_number = mol.atomic_number

        # define the BAS positions
        self.nshells = torch.tensor(mol.nshells)
        self.bas_coords = self.atom_coords.repeat_interleave(
            self.nshells, dim=0)
        self.nbas = len(self.bas_coords)

        # index for the contractions
        self.index_ctr = torch.tensor(mol.index_ctr)

        # get the coeffs of the bas
        self.bas_coeffs = torch.tensor(mol.bas_coeffs)

        # get the exponents of the bas
        self.bas_exp = nn.Parameter(torch.tensor(mol.bas_exp))
        self.bas_exp.requires_grad = True

        # get the quantum number
        self.bas_n = torch.tensor(mol.bas_n).type(torch.get_default_dtype())
        self.bas_l = torch.tensor(mol.bas_l)
        self.bas_m = torch.tensor(mol.bas_m)

        # select the radial aprt
        radial_dict = {'sto': self._radial_slater,
                       'gto': self._radial_gaussian}
        self.radial = radial_dict[mol.basis_type]

        # get the normaliationconstants
        self.norm_cst = self.get_norm(mol.basis_type)

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self._to_device()

    def get_norm(self, basis_type):
        """Compute the normalization factor of the atomic orbitals.

        Arguments:
            basis_type {str]} -- basis type 'sto' or 'gto'

        Returns:
            torch.tensor -- normalization factor
        """

        with torch.no_grad():

            if basis_type == 'sto':
                return self._norm_slater()

            elif basis_type == 'gto':
                return self._norm_gaussian()

    def _norm_slater(self):
        """ Normalization of STOs
        [1] www.theochem.ru.nl/~pwormer/Knowino/knowino.org/wiki/Slater_orbital.html
        [2] C Filippi, Multiconf wave functions for QMC of first row diatomic molecules, 
            JCP 105, 213 1996
        [3] Monte Carlo Methods in Ab Inition Quantum Chemistry, B.L. Hammond

        Returns:
            torch.tensor -- normalization factor
        """
        nfact = torch.tensor([np.math.factorial(2*n)
                              for n in self.bas_n], dtype=torch.get_default_dtype())
        return (2*self.bas_exp)**self.bas_n * torch.sqrt(2*self.bas_exp / nfact)

    def _norm_gaussian(self):
        """ Normlization of GTOs.
        [1] Computational Quantum Chemistry: An interactive Intrduction to basis set theory 
            eq : 1.14 page 23.'''

        Returns:
            torch.tensor -- normalization factor
        """

        from scipy.special import factorial2 as f2

        bas_n = self.bas_n+1.
        exp1 = 0.25*(2.*bas_n+1.)

        A = self.bas_exp**exp1
        B = 2**(2.*bas_n+3./2)
        C = torch.tensor(f2(2*bas_n.int()-1)*np.pi **
                         0.5).type(torch.get_default_dtype())

        return torch.sqrt(B/C)*A

    def _radial_slater(self, R, xyz=None, derivative=0, jacobian=True):
        """Compute the radial part of STOs (or its derivative).

        Arguments:
            R {torch.tensor} -- distance between each electron and each atom

        Keyword Arguments:
            xyz {torch.tensor} -- positions of the electrons (needed for derivative) (default: {None})
            derivative {int} -- degree of the derivative (default: {0})
            jacobian {bool} -- return the jacobian, i.e the sum of the gradients (default: {True})

        Returns:
            torch.tensor -- values of each orbital radial part at each position
        """

        if derivative == 0:
            return R**self.bas_n * torch.exp(-self.bas_exp*R)

        elif derivative > 0:

            rn = R**(self.bas_n)
            nabla_rn = (self.bas_n * R**(self.bas_n-2)).unsqueeze(-1) * xyz

            er = torch.exp(-self.bas_exp*R)
            nabla_er = -(self.bas_exp * er).unsqueeze(-1) * \
                xyz / R.unsqueeze(-1)

            if derivative == 1:

                if jacobian:
                    nabla_rn = nabla_rn.sum(3)
                    nabla_er = nabla_er.sum(3)
                    return nabla_rn*er + rn*nabla_er
                else:
                    return nabla_rn*er.unsqueeze(-1) + rn.unsqueeze(-1)*nabla_er

            elif derivative == 2:

                sum_xyz2 = (xyz**2).sum(3)

                lap_rn = self.bas_n * (3*R**(self.bas_n-2)
                                       + sum_xyz2 * (self.bas_n-2) * R**(self.bas_n-4))

                lap_er = self.bas_exp**2 * er * sum_xyz2 / R**2 \
                    - 2 * self.bas_exp * er * sum_xyz2 / R**3

                return lap_rn*er + 2*(nabla_rn*nabla_er).sum(3) + rn*lap_er

    def _radial_gaussian(self, R, xyz=None, derivative=0, jacobian=True):
        """Compute the radial part of GTOs (or its derivative).

        Arguments:
            R {torch.tensor} -- distance between each electron and each atom

        Keyword Arguments:
            xyz {torch.tensor} -- positions of the electrons (needed for derivative) (default: {None})
            derivative {int} -- degree of the derivative (default: {0})
            jacobian {bool} -- return the jacobian, i.e the sum of the gradients (default: {True})

        Returns:
            torch.tensor -- values of each orbital radial part at each position
        """
        if derivative == 0:
            return R**self.bas_n * torch.exp(-self.bas_exp*R**2)

        elif derivative > 0:

            rn = R**(self.bas_n)
            nabla_rn = (self.bas_n * R**(self.bas_n-2)).unsqueeze(-1) * xyz

            er = torch.exp(-self.bas_exp*R**2)
            nabla_er = -2*(self.bas_exp * er).unsqueeze(-1) * xyz

            if derivative == 1:
                if jacobian:
                    nabla_rn = nabla_rn.sum(3)
                    nabla_er = nabla_er.sum(3)
                    return nabla_rn*er + rn*nabla_er
                else:
                    return nabla_rn*er.unsqueeze(-1) + rn.unsqueeze(-1)*nabla_er

            elif derivative == 2:

                lap_rn = self.bas_n * (3*R**(self.bas_n-2)
                                       + (xyz**2).sum(3) * (self.bas_n-2) * R**(self.bas_n-4))

                lap_er = 4 * self.bas_exp**2 * (xyz**2).sum(3) * er \
                    - 6 * self.bas_exp * er

                return lap_rn*er + 2*(nabla_rn*nabla_er).sum(3) + rn*lap_er

    def _to_device(self):
        """Export the non parameter variable to the device."""

        self.device = torch.device('cuda')
        self.to(self.device)
        attrs = ['bas_n', 'bas_l', 'bas_m', 'bas_coeffs',
                 'nshells', 'norm_cst', 'index_ctr']
        for at in attrs:
            self.__dict__[at] = self.__dict__[at].to(self.device)

    def forward(self, input, derivative=0, jacobian=True, one_elec=False):
        """Computes the values of the atomic orbitals (or their derivatives)
        for the electrons positions in input.

        Args:
            input (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0,1,2,).
                                        Defaults to 0.
            jacobian (bool, optional): Return the jacobian (i.e. the sum of
                                       the derivatives) or the individual
                                       terms. Defaults to True.
                                       False only for derivative=1

            one_elec (bool, optional): if only one electron is in input

        Returns:
            torch.tensor: Value of the AO (or their derivatives)
                          size : Nbatch, Nelec, Norb (jacobian = True)
                          size : Nbatch, Nelec, Norb, Ndim (jacobian = False)
        """

        if not jacobian:
            assert(derivative == 1)

        if one_elec:
            nelec_save = self.nelec
            self.nelec = 1

        nbatch = input.shape[0]

        # get the pos of the bas
        self.bas_coords = self.atom_coords.repeat_interleave(
            self.nshells, dim=0)

        # get the x,y,z, distance component of each point from each RBF center
        # -> (Nbatch,Nelec,Nbas,Ndim)
        xyz = (input.view(-1, self.nelec, 1, self.ndim) -
               self.bas_coords[None, ...])
        # print('xyz : ', time()-t0)

        # compute the distance
        # -> (Nbatch,Nelec,Nbas)
        r = torch.sqrt((xyz**2).sum(3))

        # radial part
        # -> (Nbatch,Nelec,Nbas)
        R = self.radial(r)

        # compute by the spherical harmonics
        # -> (Nbatch,Nelec,Nbas)
        Y = SphericalHarmonics(xyz, self.bas_l, self.bas_m)

        # values of AO
        # -> (Nbatch,Nelec,Nbas)
        if derivative == 0:
            bas = R * Y

        # values of first derivative
        elif derivative == 1:

            # return the jacobian
            if jacobian:
                dR = self.radial(r, xyz=xyz, derivative=1)
                dY = SphericalHarmonics(
                    xyz, self.bas_l, self.bas_m, derivative=1)
                # -> (Nbatch,Nelec,Nbas)
                bas = dR * Y + R * dY

            # returm individual components
            else:
                dR = self.radial(r, xyz=xyz, derivative=1, jacobian=False)
                dY = GradSphericalHarmonics(xyz, self.bas_l, self.bas_m)
                # -> (Nbatch,Nelec,Nbas,Ndim)
                bas = dR * Y.unsqueeze(-1) + R.unsqueeze(-1) * dY

        # second derivative
        elif derivative == 2:
            dR = self.radial(r, xyz=xyz, derivative=1, jacobian=False)
            dY = GradSphericalHarmonics(xyz, self.bas_l, self.bas_m)

            d2R = self.radial(r, xyz=xyz, derivative=2)
            d2Y = SphericalHarmonics(xyz, self.bas_l, self.bas_m, derivative=2)

            bas = d2R * Y + 2. * (dR * dY).sum(3) + R * d2Y

        # product with coefficients and primitives norm
        if jacobian:

            # -> (Nbatch,Nelec,Nbas)
            bas = self.norm_cst * self.bas_coeffs * bas

            # contract the basis
            # -> (Nbatch,Nelec,Norb)
            ao = torch.zeros(nbatch, self.nelec, self.norb, device=self.device)
            ao.index_add_(2, self.index_ctr, bas)

        else:
            # -> (Nbatch,Nelec,Nbas, Ndim)
            bas = self.norm_cst.unsqueeze(-1) * \
                self.bas_coeffs.unsqueeze(-1) * bas

            # contract the basis
            # -> (Nbatch,Nelec,Norb, Ndim)
            ao = torch.zeros(nbatch, self.nelec, self.norb,
                             3, device=self.device)
            ao.index_add_(2, self.index_ctr, bas)

        if one_elec:
            self.nelec = nelec_save

        return ao

    def update(self, ao, pos, idelec):
        """Update the AO matrix if only the idelec electron has been moved.

        Arguments:
            ao {torch.tensor} -- input ao matrix
            pos {torch.tensor} -- position of the electron that has moved
            idelec {int} -- index of the electron that has moved

        Returns:
            torch.tensor -- new ao matrix
        """
        ao_new = ao.clone()
        ids, ide = (idelec)*3, (idelec+1)*3
        ao_new[:, idelec, :] = self.forward(
            pos[:, ids:ide], one_elec=True).squeeze(1)
        return ao_new


if __name__ == "__main__":

    from deepqmc.wavefunction.molecule import Molecule
    from time import time
    m = Molecule(atom='C 0 0 0; O 0 0 3.015',
                 basis_type='gto', basis='sto-6g')

    ao = AtomicOrbitals(m, cuda=False)

    pos = torch.rand(10, ao.nelec*3)

    t0 = time()
    aoval = ao(pos)
    print('Total calculation : ', time()-t0)

    t0 = time()
    aoval = ao(pos[:, :3], one_elec=True)
    print('1elec, calculation : ', time()-t0)
