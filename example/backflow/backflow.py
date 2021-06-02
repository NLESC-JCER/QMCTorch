import torch
from torch import nn

from qmctorch.scf import Molecule
from qmctorch.wavefunction.orbitals.backflow.kernels import BackFlowKernelBase
from qmctorch.wavefunction import SlaterJastrowBackFlow


class MyBackflow(BackFlowKernelBase):

    def __init__(self, mol, cuda, size=16):
        super().__init__(mol, cuda)
        self.fc1 = nn.Linear(1, size, bias=False)
        self.fc2 = nn.Linear(size, 1, bias=False)

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, 1)
        x = self.fc2(self.fc1(x))
        return x.reshape(*original_shape)


# define the molecule
mol = Molecule(atom='Li 0. 0. 0.; H 3.14 0. 0.', unit='angs',
               calculator='pyscf', basis='sto-3g', name='water')

# define the wave function
wf = SlaterJastrowBackFlow(mol, kinetic='jacobi',
                           backflow_kernel=MyBackflow,
                           backflow_kernel_kwargs={'size': 64},
                           configs='single_double(2,2)')

pos = torch.rand(10, wf.nelec*3)
print(wf(pos))
