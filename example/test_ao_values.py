import sys
import torch
from torch.autograd import Variable
from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.wavefunction.molecule import Molecule
from pyscf import gto

import matplotlib.pyplot as plt

# define the molecule
at = 'N 0 0 -1; N 0 0 1'
mol = Molecule(atom=at, 
               basis_type='gto', 
               basis='sto-3g', 
               unit='bohr')


# define the wave function
wf = Orbital(mol)

# solver
pos = torch.zeros(100,mol.nelec*3)
pos[:,2] = torch.linspace(-5,5,100)

pos = Variable(pos)
pos.requires_grad = True
aovals = wf.ao(pos)


# pyscf
m = gto.M(atom=at, basis='sto-3g',unit='bohr')
aovals_ref = m.eval_gto('GTOval_cart',pos.detach().numpy()[:,:3])

norb = 2
x = pos[:,2].detach().numpy()
plt.plot(x,aovals[:,0,3].detach().numpy(),label='torch')
plt.plot(x,aovals_ref[:,4],label='pyscf')
plt.legend()
plt.show()

#pos = solver.single_point()

# plot the molecule
#plot_molecule(solver)

# optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')

# plot the data
# plot_observable(solver.obs_dict,e0=-1.16)











