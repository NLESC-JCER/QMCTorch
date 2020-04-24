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
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        raise NotImplementedError()

    def nuclear_potential(self, pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Ven * psi
        '''
        raise NotImplementedError()

    def nuclear_repulsion(self):
        '''Compute the nuclear repulsion term

        Returns: values of Vnn * psi
        '''
        raise NotImplementedError()

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
                       # create_graph is REQUIRED and
                       # is causing memory issues
                       # for large systems
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
