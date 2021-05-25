
import torch
from torch import nn
from functools import reduce
from .elec_elec.kernels.pade_jastrow_kernel import PadeJastrowKernel
from .elec_elec.jastrow_factor_electron_electron import JastrowFactorElectronElectron
from .elec_nuclei.jastrow_factor_electron_nuclei import JastrowFactorElectronNuclei
from .elec_elec_nuclei.jastrow_factor_electron_electron_nuclei import JastrowFactorElectronElectronNuclei


class JastrowFactorCombinedTerms(nn.Module):

    def __init__(self, nup, ndown, atomic_pos,
                 jastrow_kernel={
                     'ee': PadeJastrowKernel,
                     'en': None,
                     'een': None},
                 jastrow_kernel_kwargs={
                     'ee': {},
                     'en': {},
                     'een': {}},
                 cuda=False):
        """[summary]

        Args:
            nup ([type]): [description]
            ndown ([type]): [description]
            atomic_pos ([type]): [description]
            elec_elec_kernel ([type], optional): [description]. Defaults to PadeJastrowKernel.
            elec_elec_kernel_kwargs (dict, optional): [description]. Defaults to {}.
            elec_nuc_kernel ([type], optional): [description]. Defaults to None.
            elec_nuc_kernel_kwargs (dict, optional): [description]. Defaults to {}.
            elec_elec_nuc_kernel ([type], optional): [description]. Defaults to None.
            elec_elec_nuc_kernel_kwargs (dict, optional): [description]. Defaults to {}.
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
                jastrow_kernel[k] = {}

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
                                                                          jastrow_kernel['en'],
                                                                          jastrow_kernel_kwargs['en'],
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

            jast_vals = [term(pos) for term in self.jastrow_terms]
            djast_vals = [term(pos, derivative=1, sum_grad=sum_grad)
                          for term in self.jastrow_terms]
            return self.get_derivative_combined_values(jast_vals, djast_vals)

        elif derivative == 2:

            jast_vals = [term(pos) for term in self.jastrow_terms]
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

            return(self.get_combined_values(jast_vals),
                   self.get_derivative_combined_values(
                       jast_vals, djast_vals),
                   self.get_second_derivative_combined_values(jast_vals, djast_vals, d2jast_vals))

        else:
            raise ValueError('derivative not understood')

    @ staticmethod
    def get_combined_values(jast_vals):
        """Compute the product of all terms in jast_vals."""
        return reduce(lambda x, y: x*y, jast_vals)

    @ staticmethod
    def get_derivative_combined_values(jast_vals, djast_vals):
        """Compute the derivative of the product.
        .. math:
            J = A * B * C
            \\frac{d J}{dx} = \\frac{d A}{dx} B C + A \\frac{d B}{dx} C + A B \\frac{d C}{dx}
        """
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
        out = 0.
        nterms = len(jast_vals)
        for i in range(nterms):

            # d2a * b * c
            tmp = jast_vals.copy()
            tmp[i] = d2jast_vals[i]
            out = out + reduce(lambda x, y: x*y, tmp)

            # da * db * c
            tmp = djast_vals.copy()
            tmp[i] = jast_vals[i].unsqueeze(-1)
            out = out + (2*reduce(lambda x, y: x*y, tmp)).sum(1)

        return out
