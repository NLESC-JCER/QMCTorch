import torch

from torch.autograd import grad, Variable


class WaveFunction(torch.nn.Module):

    def __init__(self, nelec, ndim, kinetic='auto', cuda=False):

        super(WaveFunction, self).__init__()

        self.ndim = ndim
        self.nelec = nelec
        self.ndim_tot = self.nelec * self.ndim
        self.kinetic = kinetic
        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

    def forward(self, x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''

        raise NotImplementedError()

    def electronic_potential(self, pos):
        r"""Computes the electron-electron term

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

    def nuclear_potential(self, pos):
        r"""Computes the electron-nuclear term

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

    def nuclear_repulsion(self):
        r"""Computes the nuclear-nuclear repulsion term

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

    def kinetic_energy(self, pos):
        '''Main switch for the kinetic energy.'''

        if self.kinetic == 'auto':
            return self.kinetic_energy_autograd(pos)

        elif self.kinetic == 'fd':
            return self.kinetic_energy_finite_difference(pos)

        else:
            raise ValueError(
                'kinetic %s not recognized' %
                self.kinetic)

    def kinetic_energy_autograd(self, pos):
        '''Compute the second derivative of the network
        output w.r.t the value of the input.

        This is to compute the value of the kinetic operator.

        Args:
            pos: position of the electron
            out : preomputed values of the wf at pos

        Returns:
            values of nabla^2 * Psi
        '''

        out = self.forward(pos)

        # compute the jacobian
        z = torch.ones_like(out)
        jacob = grad(out, pos,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]

        # compute the diagonal element of the Hessian
        z = Variable(torch.ones(jacob.shape[0])).to(self.device)
        hess = torch.zeros(jacob.shape[0]).to(self.device)

        for idim in range(jacob.shape[1]):

            tmp = grad(jacob[:, idim], pos,
                       grad_outputs=z,
                       only_inputs=True,
                       create_graph=True)[0]

            hess += tmp[:, idim]

        return -0.5 * hess.view(-1, 1) / out

    def kinetic_energy_finite_difference(self, pos, eps=1E-3):
        '''Compute the second derivative of the network
        output w.r.t the value of the input using finite difference.

        This is to compute the value of the kinetic operator.

        Args:
            pos: position of the electron
            out : preomputed values of the wf at pos
            eps : psilon for numerical derivative
        Returns:
            values of nabla^2 * Psi
        '''

        nwalk = pos.shape[0]
        ndim = pos.shape[1]
        out = torch.zeros(nwalk, 1)

        pos_tmp = pos.clone()
        out = -2 * self.forward(pos_tmp)

        for icol in range(ndim):

            feps = out.clone()

            pos_tmp = pos.clone()
            pos_tmp[:, icol] += eps
            feps += self.forward(pos_tmp)

            pos_tmp = pos.clone()
            pos_tmp[:, icol] -= eps
            feps += self.forward(pos_tmp)

            out += feps / (eps**2)

        return -0.5 * out.view(-1, 1) / out

    def local_energy(self, pos):
        ''' local energy of the sampling points.'''

        ke = self.kinetic_energy(pos)

        return ke  \
            + self.nuclear_potential(pos)  \
            + self.electronic_potential(pos) \
            + self.nuclear_repulsion()

    def energy(self, pos):
        '''Total energy for the sampling points.'''
        return torch.mean(self.local_energy(pos))

    def variance(self, pos):
        '''Variance of the energy at the sampling points.'''
        return torch.var(self.local_energy(pos))

    def sampling_error(self, eloc):
        '''Compute the statistical uncertainty.
        Assuming the samples are uncorrelated.'''
        Npts = eloc.shape[0]
        return torch.sqrt(eloc.var() / Npts)

    def _energy_variance(self, pos):
        '''Return energy and variance.'''
        el = self.local_energy(pos)
        return torch.mean(el), torch.var(el)

    def _energy_variance_error(self, pos):
        '''Return energy variance and sampling error.'''
        el = self.local_energy(pos)
        return torch.mean(el), torch.var(el), self.sampling_error(el)

    def pdf(self, pos):
        '''density of the wave function.'''
        return (self.forward(pos)**2).reshape(-1)

    def get_number_parameters(self):
        """Computes the total number of parameters."""
        nparam = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                nparam += param.data.numel()
        return nparam
