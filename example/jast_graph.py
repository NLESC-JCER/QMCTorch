
from qmctorch.wavefunction.jastrows.graph.jastrow_graph import JastrowFactorGraph
import torch
from torch.autograd import grad
nup = 2
ndown = 2
atomic_pos = torch.rand(2, 3)
atom_types = ["Li", "H"]
jast = JastrowFactorGraph(nup, ndown,
                          atomic_pos,
                          atom_types)


pos = torch.rand(10, 12)
pos.requires_grad = True
jval = jast(pos)

gval = jast(pos, derivative=1)
hval = jast(pos, derivative=2)
