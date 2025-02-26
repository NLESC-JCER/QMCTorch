
from qmctorch.wavefunction.jastrows.graph.jastrow_graph import MGCNJastrowFactor
import torch
from torch.autograd import grad
from types import SimpleNamespace
# from qmctorch.wavefunction.jastrows.graph.mgcn.mgcn_predictor import MGCNPredictor
from dgllife.model.model_zoo.mgcn_predictor import MGCNPredictor

nup = 2
ndown = 2
atomic_pos = torch.rand(2, 3)
atom_types = ["Li", "H"]

mol = SimpleNamespace(
    nup=nup,
    ndown=ndown,
    atom_coords=atomic_pos,
    atoms=atom_types,
)

jast = MGCNJastrowFactor(
            mol,
            ee_model_kwargs={"n_layers": 3, "feats": 32, "cutoff": 5.0, "gap": 1.0},
            en_model_kwargs={"n_layers": 3, "feats": 32, "cutoff": 5.0, "gap": 1.0},
        )


pos = torch.rand(10, 12)
pos.requires_grad = True
jval = jast(pos)

gval = jast(pos, derivative=1)
hval = jast(pos, derivative=2)
