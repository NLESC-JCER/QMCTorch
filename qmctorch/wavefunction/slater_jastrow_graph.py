

import numpy as np
import torch
from .slater_jastrow import SlaterJastrow

from .jastrows.elec_elec.kernels.pade_jastrow_kernel import PadeJastrowKernel
from .jastrows.elec_elec.jastrow_factor_electron_electron import JastrowFactorElectronElectron

from .jastrows.graph.jastrow_graph import JastrowFactorGraph
from .jastrows.graph.mgcn.mgcn_predictor import MGCNPredictor


class SlaterJastrowGraph(SlaterJastrow):

    def __init__(self, mol, configs='ground_state',
                 kinetic='auto',
                 ee_model=MGCNPredictor,
                 ee_model_kwargs={},
                 en_model=MGCNPredictor,
                 en_model_kwargs={},
                 atomic_features=["atomic_number"],
                 cuda=False,
                 include_all_mo=True):
        """Implementation of a SlaterJastrow Network using Graph neural network to express the Jastrow.

        Args:
            mol (qmc.wavefunction.Molecule): a molecule object
            configs (str, optional): defines the CI configurations to be used. Defaults to 'ground_state'.
            kinetic (str, optional): method to compute the kinetic energy. Defaults to 'jacobi'.
            ee_network (dgl model): graph network of the elec-elec factor
            ee_network_kwargs (dict, optional): Argument of the elec-elec graph network. Defaults to {}.
            en_network (dgl model): graph network of the elec-nuc factor
            en_network_kwargs (dict, optional): Argument of the elec-nuc graph network. Defaults to {}.
            atomic_featires (list, optional): list of atomic properties from medeleev
            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
            include_all_mo (bool, optional): include either all molecular orbitals or only the ones that are
                                             popualted in the configs. Defaults to False
        Examples::
            >>> mol = Molecule('h2o.xyz', calculator='adf', basis = 'dzp')
            >>> wf = SlaterJastrow(mol, configs='cas(2,2)')
        """
        if kinetic != "auto":
            raise ValueError("Kinetic energy must be set to auto")

        super().__init__(mol, configs, kinetic, None, None, cuda, include_all_mo)

        ee_name = ee_model.__name__
        en_name =  en_model.__name__ if en_model is not None else 'None'

        self.jastrow_type = 'Graph(ee:%s, en:%s)' % (
           ee_name , en_name)
        self.use_jastrow = True
        self.jastrow = JastrowFactorGraph(mol.nup, mol.ndown,
                                          torch.as_tensor(
                                              mol.atom_coords),
                                          mol.atoms,
                                          ee_model=ee_model,
                                          ee_model_kwargs=ee_model_kwargs,
                                          en_model=en_model,
                                          en_model_kwargs=en_model_kwargs,
                                          atomic_features=atomic_features,
                                          cuda=cuda)

        if self.cuda:
            self.jastrow = self.jastrow.to(self.device)

        self.log_data()
