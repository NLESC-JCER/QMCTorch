
import torch
from torch import nn
from functools import reduce

from .elec_elec.jastrow_factor_electron_electron import JastrowFactorElectronElectron
from .elec_nuclei.jastrow_factor_electron_nuclei import JastrowFactorElectronNuclei
from .elec_elec_nuclei.jastrow_factor_electron_electron_nuclei import JastrowFactorElectronElectronNuclei


from .elec_elec.kernels.pade_jastrow_kernel import PadeJastrowKernel as PadeJastrowKernelElecElec
from .elec_nuclei.kernels.pade_jastrow_kernel import PadeJastrowKernel as PadeJastrowKernelElecNuc


class JastrowFactorCombinedTerms(nn.Module):

    def __init__(self, nup, ndown, atomic_pos,
                 jastrow_kernel={
                     'ee': PadeJastrowKernelElecElec,
                     'en': PadeJastrowKernelElecNuc,
                     'een': None},
                 jastrow_kernel_kwargs={
                     'ee': {},
                     'en': {},
                     'een': {}},
                 cuda=False):
        """[summary]

        Args:
            nup (int): number of spin up electron
            ndown (int): number opf spin down electron
            atomic_pos (torch tensor): atomic positions
            jastrow_kernel ([dict]): kernels of the jastrow factor
            jastrow_kernel_kwargs (dict): keyword argument of the kernels
            cuda (bool, optional): [description]. Defaults to False.
        """

        super().__init__()
        self.nup = nup
        self.ndown = ndown
        self.cuda = cuda

        self.jastrow_terms = []

        # sanitize the dict
        for k in ['ee', 'en', 'een']:
            if k not in jastrow_kernel.keys():
                jastrow_kernel[k] = None
            if k not in jastrow_kernel_kwargs.keys():
                jastrow_kernel_kwargs[k] = {}

        self.requires_autograd = True

        if jastrow_kernel['ee'] is not None:

            self.jastrow_terms.append(JastrowFactorElectronElectron(nup, ndown,
                                                                    jastrow_kernel['ee'],
                                                                    jastrow_kernel_kwargs['ee'],
                                                                    cuda=cuda))

        if jastrow_kernel['en'] is not None:

            self.jastrow_terms.append(JastrowFactorElectronNuclei(nup, ndown,
                                                                  atomic_pos,
                                                                  jastrow_kernel['en'],
                                                                  jastrow_kernel_kwargs['en'],
                                                                  cuda=cuda))

        if jastrow_kernel['een'] is not None:

            self.jastrow_terms.append(JastrowFactorElectronElectronNuclei(nup, ndown,
                                                                          atomic_pos,
                                                                          jastrow_kernel['een'],
                                                                          jastrow_kernel_kwargs['een'],
                                                                          cuda=cuda))
        self.nterms = len(self.jastrow_terms)

    def forward(self, pos, derivative=0, sum_grad=True):
        """Compute the Jastrow factors.

        Args:
            pos(torch.tensor): Positions of the electrons
               Size: Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0, 1, 2,).
              Defaults to 0.
            sum_grad(bool, optional): Return the sum_grad(i.e. the sum of
                                                           the derivatives)
              terms. Defaults to True.
                False only for derivative = 1

        Returns:
            torch.tensor: value of the jastrow parameter for all confs
              derivative = 0  (Nmo) x Nbatch x 1
               derivative = 1  (Nmo) x Nbatch x Nelec
                  (for sum_grad = True)
                derivative = 1  (Nmo) x Nbatch x Ndim x Nelec
                  (for sum_grad = False)
        """
        if derivative == 0:

            jast_vals = [term(pos) for term in self.jastrow_terms]
            return self.get_combined_values(jast_vals)

        elif derivative == 1:

            if sum_grad:
                jast_vals = [term(pos) for term in self.jastrow_terms]
            else:
                jast_vals = [term(pos).unsqueeze(-1)
                             for term in self.jastrow_terms]
            djast_vals = [term(pos, derivative=1, sum_grad=sum_grad)
                          for term in self.jastrow_terms]

            return self.get_derivative_combined_values(jast_vals, djast_vals)

        elif derivative == 2:

            jast_vals = [term(pos)
                         for term in self.jastrow_terms]
            djast_vals = [term(pos, derivative=1, sum_grad=False)
                          for term in self.jastrow_terms]
            d2jast_vals = [term(pos, derivative=2)
                           for term in self.jastrow_terms]
            return self.get_second_derivative_combined_values(jast_vals, djast_vals, d2jast_vals)

        elif derivative == [0, 1, 2]:

            jast_vals = [term(pos) for term in self.jastrow_terms]
            djast_vals = [term(pos, derivative=1, sum_grad=False)
                          for term in self.jastrow_terms]
            d2jast_vals = [term(pos, derivative=2)
                           for term in self.jastrow_terms]

            # combine the jastrow terms
            out_jast = self.get_combined_values(jast_vals)

            # combine the second derivative
            out_d2jast = self.get_second_derivative_combined_values(
                jast_vals, djast_vals, d2jast_vals)

            # unsqueeze the jast terms to be compatible with the
            # derivative
            jast_vals = [j.unsqueeze(-1) for j in jast_vals]

            # combine the derivative
            out_djast = self.get_derivative_combined_values(
                jast_vals, djast_vals)

            return(out_jast, out_djast, out_d2jast)

        else:
            raise ValueError('derivative not understood')

    @ staticmethod
    def get_combined_values(jast_vals):
        """Compute the product of all terms in jast_vals."""
        if len(jast_vals) == 1:
            return jast_vals[0]
        else:
            return reduce(lambda x, y: x*y, jast_vals)

    @ staticmethod
    def get_derivative_combined_values(jast_vals, djast_vals):
        """Compute the derivative of the product.
        .. math:
            J = A * B * C
            \\frac{d J}{dx} = \\frac{d A}{dx} B C + A \\frac{d B}{dx} C + A B \\frac{d C}{dx}
        """
        if len(djast_vals) == 1:
            return djast_vals[0]
        else:
            out = 0.
            nterms = len(jast_vals)
            for i in range(nterms):
                tmp = jast_vals.copy()
                tmp[i] = djast_vals[i]
                out += reduce(lambda x, y: x*y, tmp)
            return out

    @ staticmethod
    def get_second_derivative_combined_values(jast_vals, djast_vals, d2jast_vals):
        """Compute the derivative of the product.
        .. math:
            J = A * B * C
            \\frac{d^2 J}{dx^2} = \\frac{d^2 A}{dx^2} B C + A \\frac{d^2 B}{dx^2} C + A B \\frac{d^2 C}{dx^2} \\
                               + 2( \\frac{d A}{dx} \\frac{dB}{dx} C + \\frac{d A}{dx} B \\frac{dC}{dx} + A \\frac{d B}{dx} \\frac{dC}{dx} )
        """
        if len(d2jast_vals) == 1:
            return d2jast_vals[0]
        else:
            out = 0.
            nterms = len(jast_vals)
            for i in range(nterms):

                # d2a * b * c
                tmp = jast_vals.copy()
                tmp[i] = d2jast_vals[i]
                out = out + reduce(lambda x, y: x*y, tmp)

            for i in range(nterms-1):
                for j in range(i+1, nterms):

                    # da * db * c
                    tmp = jast_vals.copy()
                    tmp = [j.unsqueeze(-1) for j in tmp]
                    tmp[i] = djast_vals[i]
                    tmp[j] = djast_vals[j]

                    out = out + \
                        (2.*reduce(lambda x, y: x*y, tmp)).sum(1)

            return out
