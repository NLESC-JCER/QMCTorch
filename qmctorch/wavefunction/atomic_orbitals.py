import torch
from torch import nn
from .radial_functions import radial_gaussian, radial_slater
from .norm_orbital import atomic_orbital_norm
from .spherical_harmonics import Harmonics
from ..utils import register_extra_attributes


class AtomicOrbitals(nn.Module):

    def __init__(self, mol, cuda=False):
        """Computes the value of atomic orbitals

        Args:
            mol (Molecule): Molecule object
            cuda (bool, optional): Turn GPU ON/OFF Defaults to False.
        """

        super(AtomicOrbitals, self).__init__()
        dtype = torch.get_default_dtype()

        # wavefunction data
        self.nelec = mol.nelec
        self.norb = mol.basis.nao
        self.ndim = 3

        # make the atomic position optmizable
        self.atom_coords = nn.Parameter(torch.tensor(
            mol.basis.atom_coords_internal).type(dtype))
        self.atom_coords.requires_grad = True
        self.natoms = len(self.atom_coords)
        self.atomic_number = mol.atomic_number

        # define the BAS positions.
        self.nshells = torch.tensor(mol.basis.nshells)
        self.bas_coords = self.atom_coords.repeat_interleave(
            self.nshells, dim=0)
        self.nbas = len(self.bas_coords)

        # index for the contractions
        self.index_ctr = torch.tensor(mol.basis.index_ctr)

        # get the coeffs of the bas
        self.bas_coeffs = torch.tensor(
            mol.basis.bas_coeffs).type(dtype)

        # get the exponents of the bas
        self.bas_exp = nn.Parameter(
            torch.tensor(mol.basis.bas_exp).type(dtype))
        self.bas_exp.requires_grad = True

        # harmonics generator
        if mol.basis.harmonics_type == 'sph':
            self.bas_n = torch.tensor(mol.basis.bas_n).type(dtype)
            self.harmonics = Harmonics(
                mol.basis.harmonics_type,
                bas_l=mol.basis.bas_l,
                bas_m=mol.basis.bas_m)

        elif mol.basis.harmonics_type == 'cart':
            self.bas_n = torch.tensor(mol.basis.bas_kr).type(dtype)
            self.harmonics = Harmonics(
                mol.basis.harmonics_type,
                bas_kx=mol.basis.bas_kx,
                bas_ky=mol.basis.bas_ky,
                bas_kz=mol.basis.bas_kz)

        # select the radial apart
        radial_dict = {'sto': radial_slater,
                       'gto': radial_gaussian}
        self.radial = radial_dict[mol.basis.radial_type]

        # get the normalisation constants
        if hasattr(mol.basis, 'bas_norm') and False:
            self.norm_cst = torch.tensor(
                mol.basis.bas_norm).type(dtype)
        else:
            with torch.no_grad():
                self.norm_cst = atomic_orbital_norm(
                    mol.basis).type(dtype)

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self._to_device()

    def _to_device(self):
        """Export the non parameter variable to the device."""

        self.device = torch.device('cuda')
        self.to(self.device)
        attrs = ['bas_n', 'bas_coeffs',
                 'nshells', 'norm_cst', 'index_ctr']
        for at in attrs:
            self.__dict__[at] = self.__dict__[at].to(self.device)

    def forward(self, input, derivative=0, jacobian=True, one_elec=False):
        """Computes the values of the atomic orbitals.

        .. math::
            \phi_i(r_j) = \sum_n c_n \\text{Rad}^{i}_n(r_j) \\text{Y}^{i}_n(r_j)

        where Rad is the radial part and Y the spherical harmonics part. 
        It is also possible to compute the first and second derivatives

        .. math::
            \\nabla \phi_i(r_j) = \\frac{d}{dx_j} \phi_i(r_j) + \\frac{d}{dy_j} \phi_i(r_j) + \\frac{d}{dz_j} \phi_i(r_j) \n
            \\text{grad} \phi_i(r_j) = (\\frac{d}{dx_j} \phi_i(r_j), \\frac{d}{dy_j} \phi_i(r_j), \\frac{d}{dz_j} \phi_i(r_j)) \n
            \Delta \phi_i(r_j) = \\frac{d^2}{dx^2_j} \phi_i(r_j) + \\frac{d^2}{dy^2_j} \phi_i(r_j) + \\frac{d^2}{dz^2_j} \phi_i(r_j)

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
            torch.tensor: Value of the AO (or their derivatives) \n
                          size : Nbatch, Nelec, Norb (jacobian = True) \n
                          size : Nbatch, Nelec, Norb, Ndim (jacobian = False)

        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> ao = AtomicOrbitals(mol)
            >>> pos = torch.rand(100,6)
            >>> aovals = ao(pos)
            >>> daovals = ao(pos,derivative=1)
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
        R = self.radial(r, self.bas_n, self.bas_exp)

        # compute by the spherical harmonics
        # -> (Nbatch,Nelec,Nbas)
        Y = self.harmonics(xyz)

        # values of AO
        # -> (Nbatch,Nelec,Nbas)
        if derivative == 0:
            bas = R * Y

        # values of first derivative
        elif derivative == 1:

            # return the jacobian
            if jacobian:
                dR = self.radial(
                    r, self.bas_n, self.bas_exp, xyz=xyz, derivative=1)
                dY = self.harmonics(xyz, derivative=1)

                # -> (Nbatch,Nelec,Nbas)
                bas = dR * Y + R * dY

            # returm individual components
            else:
                dR = self.radial(
                    r, self.bas_n, self.bas_exp, xyz=xyz, derivative=1, jacobian=False)
                dY = self.harmonics(xyz, derivative=1, jacobian=False)
                # -> (Nbatch,Nelec,Nbas,Ndim)
                bas = dR * Y.unsqueeze(-1) + R.unsqueeze(-1) * dY

        # second derivative
        elif derivative == 2:

            dR = self.radial(r, self.bas_n, self.bas_exp,
                             xyz=xyz, derivative=1, jacobian=False)
            dY = self.harmonics(xyz, derivative=1, jacobian=False)

            d2R = self.radial(
                r, self.bas_n, self.bas_exp, xyz=xyz, derivative=2)
            d2Y = self.harmonics(xyz, derivative=2)

            bas = d2R * Y + 2. * (dR * dY).sum(3) + R * d2Y

        # product with coefficients and primitives norm
        if jacobian:

            # -> (Nbatch,Nelec,Nbas)
            bas = self.norm_cst * self.bas_coeffs * bas

            # contract the basis
            # -> (Nbatch,Nelec,Norb)
            ao = torch.zeros(nbatch, self.nelec,
                             self.norb, device=self.device)
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
        """Update an AO matrix with the new positions of one electron

        Args:
            ao (torch.tensor): initial AO matrix
            pos (torch.tensor): new positions of some electrons
            idelec (int): index of the electron that has moved

        Returns:
            torch.tensor: new AO matrix

        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> ao = AtomicOrbitals(mol)
            >>> pos = torch.rand(100,6)
            >>> aovals = ao(pos)
            >>> id = 0
            >>> pos[:,:3] = torch.rand(100,3)
            >>> ao.update(aovals, pos, 0)
        """

        ao_new = ao.clone()
        ids, ide = (idelec) * 3, (idelec + 1) * 3
        ao_new[:, idelec, :] = self.forward(
            pos[:, ids:ide], one_elec=True).squeeze(1)
        return ao_new
