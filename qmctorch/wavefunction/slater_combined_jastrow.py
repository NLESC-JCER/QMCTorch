

import numpy as np
import torch
from .slater_jastrow import SlaterJastrow

from .jastrows.elec_elec.kernels.pade_jastrow_kernel import PadeJastrowKernel as PadeJastrowKernelElecElec
from .jastrows.jastrow_factor_combined_terms import JastrowFactorCombinedTerms
from .jastrows.elec_nuclei.kernels.pade_jastrow_kernel import PadeJastrowKernel as PadeJastrowKernelElecNuc


class SlaterManyBodyJastrow(SlaterJastrow):

    def __init__(self, mol, configs='ground_state',
                 kinetic='jacobi',
                 jastrow_kernel={
                     'ee': PadeJastrowKernelElecElec,
                     'en': PadeJastrowKernelElecNuc,
                     'een': None},
                 jastrow_kernel_kwargs={
                     'ee': {},
                     'en': {},
                     'een': {}},
                 cuda=False,
                 include_all_mo=True):
        """Slater Jastrow wave function with many body Jastrow factor

        .. math::
            \\Psi(R_{at}, r) = J(r)\\sum_n c_n D^\\uparrow_n(r^\\uparrow)D^\\downarrow_n(r^\\downarrow)

        with

        .. math::
            J(r) = \\exp\\left( K_{ee}(r) + K_{en}(R_{at},r) + K_{een}(R_{at}, r) \\right)    

        with the different kernels representing electron-electron, electron-nuclei and electron-electron-nuclei terms

        Args:
            mol (Molecule): a QMCTorch molecule object
            configs (str, optional): defines the CI configurations to be used. Defaults to 'ground_state'.
                - ground_state : only the ground state determinant in the wave function
                - single(n,m) : only single excitation with n electrons and m orbitals 
                - single_double(n,m) : single and double excitation with n electrons and m orbitals
                - cas(n, m) : all possible configuration using n eletrons and m orbitals                   
            kinetic (str, optional): method to compute the kinetic energy. Defaults to 'jacobi'.
                - jacobi : use the Jacobi formula to compute the kinetic energy 
                - auto : use automatic differentiation to compute the kinetic energy
            jastrow_kernel (dict, optional) : different Jastrow kernels for the different terms. 
                By default only electron-electron and electron-nuclei terms are used
            jastrow_kernel_kwargs (dict, optional) : keyword arguments for the jastrow kernels contructor
            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
            include_all_mo (bool, optional): include either all molecular orbitals or only the ones that are
                                             popualted in the configs. Defaults to False
        Examples::
            >>> from qmctorch.scf import Molecule
            >>> from qmctorch.wavefunction import SlaterManyBodyJastrow
            >>> mol = Molecule('h2o.xyz', calculator='adf', basis = 'dzp')
            >>> wf = SlaterManyBodyJastrow(mol, configs='cas(2,2)')
        """

        super().__init__(mol, configs, kinetic, None, {}, cuda, include_all_mo)

        # process the Jastrow
        if jastrow_kernel is not None:

            for k in ['ee', 'en', 'een']:
                if k not in jastrow_kernel.keys():
                    jastrow_kernel[k] = None
                if k not in jastrow_kernel_kwargs.keys():
                    jastrow_kernel_kwargs[k] = None

            self.use_jastrow = True
            self.jastrow_type = 'JastrowFactorCombinedTerms'

            self.jastrow = JastrowFactorCombinedTerms(
                self.mol.nup, self.mol.ndown,
                torch.as_tensor(self.mol.atom_coords),
                jastrow_kernel=jastrow_kernel,
                jastrow_kernel_kwargs=jastrow_kernel_kwargs,
                cuda=cuda)

            if self.cuda:
                for term in self.jastrow.jastrow_terms:
                    term = term.to(self.device)

        self.log_data()
