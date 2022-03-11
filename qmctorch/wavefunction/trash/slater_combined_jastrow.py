

import numpy as np
import torch
from .slater_jastrow import SlaterJastrow

from .jastrows.elec_elec.kernels.pade_jastrow_kernel import PadeJastrowKernel as PadeJastrowKernelElecElec
from .jastrows.jastrow_factor_combined_terms import JastrowFactorCombinedTerms
from .jastrows.elec_nuclei.kernels.pade_jastrow_kernel import PadeJastrowKernel as PadeJastrowKernelElecNuc


class SlaterCombinedJastrow(SlaterJastrow):

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
        """Implementation of the QMC Network.

        Args:
            mol (qmc.wavefunction.Molecule): a molecule object
            configs (str, optional): defines the CI configurations to be used. Defaults to 'ground_state'.
            kinetic (str, optional): method to compute the kinetic energy. Defaults to 'jacobi'.
            jastrow_kernel (JastrowKernelBase, optional) : Class that computes the jastrow kernels
            jastrow_kernel_kwargs (dict, optional) : keyword arguments for the jastrow kernel contructor
            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
            include_all_mo (bool, optional): include either all molecular orbitals or only the ones that are
                                             popualted in the configs. Defaults to False
        Examples::
            >>> mol = Molecule('h2o.xyz', calculator='adf', basis = 'dzp')
            >>> wf = SlaterJastrow(mol, configs='cas(2,2)')
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
