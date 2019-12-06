import torch
from torch.autograd import Variable
from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.wavefunction.molecule import Molecule
from pyscf import gto

import numpy as np
import matplotlib.pyplot as plt

# define the molecule
at = 'H 0 0 0; H 0 0 1'
mol = Molecule(atom=at,
               basis_type='gto',
               basis='sto-3g',
               unit='bohr')

m = gto.M(atom=at, basis='sto-3g', unit='bohr')

# define the wave function
wf = Orbital(mol)

# solver
pos = torch.zeros(100, mol.nelec*3)
pos[:, 2] = torch.linspace(-5, 5, 100)

pos = Variable(pos)
pos.requires_grad = True


aovals = wf.ao(pos)
aovals_ref = m.eval_gto('GTOval_cart', pos.detach().numpy()[:, :3])


ip_aovals = wf.ao(pos, derivative=1)
ip_aovals_ref = m.eval_gto('GTOval_ip_cart', pos.detach().numpy()[:, :3])
ip_aovals_ref = ip_aovals_ref.sum(0)

i2p_aovals = wf.ao(pos, derivative=2)

norb = 0
x = pos[:, 2].detach().numpy()

plt.plot(x, aovals[:, 0, norb].detach().numpy(), label='torch')
plt.plot(x, aovals_ref[:, norb], '-o', label='pyscf')


plt.plot(x, ip_aovals[:, 0, norb].detach().numpy(), label='torch')
plt.plot(x, ip_aovals_ref[:, norb], '-o', label='pyscf')

plt.plot(x, i2p_aovals[:, 0, norb].detach().numpy(), label='torch')
plt.plot(x, np.gradient(ip_aovals_ref[:, norb], x), label='pyscf')

plt.legend()
plt.show()

# pos = solver.single_point()

# plot the molecule
# plot_molecule(solver)

# optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')

# plot the data
# plot_observable(solver.obs_dict,e0=-1.16)
