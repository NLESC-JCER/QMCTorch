
import torch
from qmctorch.wavefunction.FermiNet_v2 import FermiNet
from qmctorch.utils import set_torch_double_precision
from qmctorch.wavefunction import Molecule

ndim = 3
set_torch_double_precision()
# define the molecule
mol = mol = Molecule(atom='O	 0.000000 0.00000  0.00000', 
            unit='bohr', calculator='pyscf')  
# network hyperparameters: 
hidden_nodes_e = 256
hidden_nodes_ee = 32
K_determinants = 4
L_layers = 4

# set a initial seed for to make the example reproducable
torch.random.manual_seed(321)
nbatch =5

# initiate a random configuration of particle positions
# r = torch.randn(nbatch,mol.nelec,ndim, device="cpu")

# using identical electron positions should if everything is correct return all 0 
r = torch.ones((nbatch,mol.nelec,ndim), device="cpu")   

WF = FermiNet(mol,hidden_nodes_e,hidden_nodes_ee,L_layers,K_determinants)
mo_up, mo_down = WF.compute_mo(r)
print(mo_up[0,0])
# check the number of parameters and layers of the Network:
# for name, param in WF.named_parameters():
#     print(name, param.size())
print("Number of parameters: {}".format(WF.get_number_parameters()))

# when using identical terms this should be close to 0
print(WF.forward(r))
