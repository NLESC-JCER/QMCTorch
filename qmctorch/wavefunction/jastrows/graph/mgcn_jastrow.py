import torch
from torch import nn
from torch.autograd import grad
from typing import Dict, List, Union, Tuple
import dgl

from dgllife.model.model_zoo.mgcn_predictor import MGCNPredictor
from ..distance.electron_electron_distance import ElectronElectronDistance
from ..distance.electron_nuclei_distance import ElectronNucleiDistance
from .elec_elec_graph import ElecElecGraph
from .elec_nuc_graph import ElecNucGraph
from ....scf import Molecule


class MGCNJastrowFactor(nn.Module):
    def __init__(
        self,
        mol: Molecule,
        ee_model_kwargs: Dict = {},
        en_model_kwargs: Dict = {},
        atomic_features: List = ["atomic_number"],
        cuda: bool = False,
    ) -> None:
        """Graph Neural Network Jastrow Factor

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            atomic_pos(torch.tensor): positions of the atoms
            atoms (list): atom type in the molecule
            ee_network (dgl model): graph network of the elec-elec factor
            ee_network_kwargs (dict, optional): Argument of the elec-elec graph network. Defaults to {}.
            en_network (dgl model): graph network of the elec-nuc factor
            en_network_kwargs (dict, optional): Argument of the elec-nuc graph network. Defaults to {}.
            atomic_featires (list, optional): list of atomic properties from medeleev
            cuda (bool, optional): use cuda. Defaults to False.
        """

        super().__init__()

        self.nup = mol.nup
        self.ndown = mol.ndown
        self.nelec = mol.nup + mol.ndown
        self.ndim = 3

        self.cuda = cuda
        self.device = torch.device("cpu")
        if self.cuda:
            self.device = torch.device("cuda")

        self.atom_types = mol.atoms
        self.atomic_features = atomic_features
        self.atoms = torch.as_tensor(mol.atom_coords).to(self.device)
        self.natoms = self.atoms.shape[0]

        self.requires_autograd = True

        # mask to extract the upper diag of the matrices
        self.mask_tri_up, self.index_col, self.index_row = self.get_mask_tri_up()

        # distance calculator
        self.elel_dist = ElectronElectronDistance(self.nelec, self.ndim)
        self.elnu_dist = ElectronNucleiDistance(self.nelec, self.atoms, self.ndim)

        # instantiate the ee mode; to use
        ee_model_kwargs["num_node_types"] = 2
        ee_model_kwargs["num_edge_types"] = 3
        self.ee_model = MGCNPredictor(**ee_model_kwargs)

        # instantiate the en model
        en_model_kwargs["num_node_types"] = 2 + self.natoms
        en_model_kwargs["num_edge_types"] = 2 * self.natoms
        self.en_model = MGCNPredictor(**en_model_kwargs)

        # compute the elec-elec graph
        self.ee_graph = ElecElecGraph(self.nelec, self.nup)

        # compute the elec-nuc graph
        self.en_graph = ElecNucGraph(
            self.natoms, self.atom_types, self.atomic_features, self.nelec, self.nup
        )

    def __repr__(self) -> str:
        """representation of the jastrow factor"""
        return "ee, en graph -> " + self.__class__.__name__

    def forward(
        self, pos: torch.Tensor, derivative: int = 0, sum_grad: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute the Jastrow factors.

        Args:
            pos (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0,1,2,).
                            Defaults to 0.
            sum_grad (bool, optional): Return the sum_grad (i.e. the sum of
                                       the derivatives) or the individual
                                       terms. Defaults to True.
                                       False only for derivative=1

        Returns:
            torch.tensor: value of the jastrow parameter for all confs
                          derivative = 0  (Nmo) x Nbatch x 1
                          derivative = 1  (Nmo) x Nbatch x Nelec (for sum_grad = True)
                          derivative = 1  (Nmo) x Nbatch x Ndim x Nelec (for sum_grad = False)
                          derivative = 2  (Nmo) x Nbatch x Nelec
        """

        size = pos.shape
        assert size[1] == self.nelec * self.ndim
        nbatch = size[0]

        batch_ee_graph = dgl.batch([self.ee_graph] * nbatch)
        batch_en_graph = dgl.batch([self.en_graph] * nbatch)

        # get the elec-elec distance matrix
        ree = self.extract_tri_up(self.elel_dist(pos)).reshape(-1, 1)

        # get the elec-nuc distance matrix
        ren = self.extract_elec_nuc_dist(self.elnu_dist(pos))

        # put the data in the graph
        batch_ee_graph.edata["distance"] = ree.repeat_interleave(2, dim=0)
        batch_en_graph.edata["distance"] = ren.repeat_interleave(2, dim=0)

        ee_node_types = batch_ee_graph.ndata.pop("node_types")
        ee_edge_distance = batch_ee_graph.edata.pop("distance")
        ee_kernel = self.ee_model(batch_ee_graph, ee_node_types, ee_edge_distance)

        en_node_types = batch_en_graph.ndata.pop("node_types")
        en_edge_distance = batch_en_graph.edata.pop("distance")
        en_kernel = self.en_model(batch_en_graph, en_node_types, en_edge_distance)

        if derivative == 0:
            return torch.exp(ee_kernel + en_kernel)

        elif derivative == 1:
            return self._get_grad_vals(pos, ee_kernel, en_kernel, sum_grad)

        elif derivative == 2:
            return self._get_hess_vals(pos, ee_kernel, en_kernel, return_all=False)

        elif derivative == [0, 1, 2]:
            return self._get_hess_vals(
                pos, ee_kernel, en_kernel, sum_grad=sum_grad, return_all=True
            )

    def _get_val(
        self, ee_kernel: torch.Tensor, en_kernel: torch.Tensor
    ) -> torch.Tensor:
        """Get the jastrow values.

        Args:
            ee_kernel ([type]): [description]
            en_kernel ([type]): [description]
        """
        return torch.exp(ee_kernel + en_kernel)

    def _get_grad_vals(
        self,
        pos: torch.Tensor,
        ee_kernel: torch.Tensor,
        en_kernel: torch.Tensor,
        sum_grad: bool,
    ) -> torch.Tensor:
        """Get the values of the gradients


        Args:
            pos ([type]): [description]
            ee_kernel ([type]): [description]
            en_kernel ([type]): [description]
            sum_grad ([type]): [description]
        """

        nbatch = len(pos)
        jval = torch.exp(ee_kernel + en_kernel)
        grad_val = grad(
            jval, pos, grad_outputs=torch.ones_like(jval), only_inputs=True
        )[0]
        grad_val = grad_val.reshape(nbatch, self.nelec, 3).transpose(1, 2)

        if sum_grad:
            grad_val = grad_val.sum(1)

        return grad_val

    def _get_hess_vals(
        self,
        pos: torch.Tensor,
        ee_kernel: torch.Tensor,
        en_kernel: torch.Tensor,
        sum_grad: bool = False,
        return_all: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get the hessian values

        Args:
            pos ([type]): [description]
            ee_kernel ([type]): [description]
            en_kernel ([type]): [description]
            sum_grad ([type]): [description]
            return_all (bool, )
        """

        nbatch = len(pos)

        jval = torch.exp(ee_kernel + en_kernel)

        grad_val = grad(
            jval,
            pos,
            grad_outputs=torch.ones_like(jval),
            only_inputs=True,
            create_graph=True,
        )[0]

        ndim = grad_val.shape[1]
        hval = torch.zeros(nbatch, ndim).to(self.device)
        z = torch.ones(grad_val.shape[0]).to(self.device)
        z.requires_grad = True

        for idim in range(ndim):
            tmp = grad(
                grad_val[:, idim],
                pos,
                grad_outputs=z,
                only_inputs=True,
                retain_graph=True,
            )[0]
            hval[:, idim] = tmp[:, idim]

        hval = hval.reshape(nbatch, self.nelec, 3).transpose(1, 2).sum(1)

        if return_all:
            grad_val = grad_val.detach().reshape(nbatch, self.nelec, 3).transpose(1, 2)

            if sum_grad:
                grad_val = grad_val.sum(1)

            return (jval, grad_val, hval)

        else:
            return hval

    def get_mask_tri_up(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Get the mask to select the triangular up matrix

        Returns:
            torch.tensor: mask of the tri up matrix
        """
        mask = torch.zeros(self.nelec, self.nelec).type(torch.bool).to(self.device)
        index_col, index_row = [], []
        for i in range(self.nelec - 1):
            for j in range(i + 1, self.nelec):
                index_row.append(i)
                index_col.append(j)
                mask[i, j] = True

        index_col = torch.LongTensor(index_col).to(self.device)
        index_row = torch.LongTensor(index_row).to(self.device)
        return mask, index_col, index_row

    def extract_tri_up(self, inp: torch.Tensor) -> torch.Tensor:
        r"""extract the upper triangular elements

        Args:
            input (torch.tensor): input matrices (..., nelec, nelec)

        Returns:
            torch.tensor: triangular up element (..., nelec_pair)
        """
        shape = list(inp.shape)
        out = inp.masked_select(self.mask_tri_up)
        return out.view(*(shape[:-2] + [-1]))

    def extract_elec_nuc_dist(self, ren: torch.Tensor) -> torch.Tensor:
        """reorganizre the elec-nuc distance to load them in the graph

        Args:
            ren (torch.tensor): distance elec-nuc [nbatch, nelec, natom]
        """
        return ren.transpose(1, 2).reshape(-1, 1)
