Creating your own Backflow transformation
==============================================

We present here how to create your own backflow transformation

During the import you must import the base class of the electron-electron Jastrow.

>>> from qmctorch.scf import Molecule
>>> from qmctorch.wavefunction import SlaterJastrow,
>>> from qmctorch.wavefunction.orbitals.backflow.kernels import BackFlowKernelBase


We can then use this base class to create a new Jastrow Factor. This is done in the same way one would create a new neural network layer in pytorch

>>> class MyBackflow(BackFlowKernelBase)
>>>
>>>     def __init__(self, mol, cuda, size=16):
>>>         super().__init__(nup, ndown, cuda)
>>>         self.fc1 = nn.Linear(1, size, bias=False)
>>>         self.fc2 = nn.Linear(size, 1, bias=False)
>>>
>>>     def forward(self, x):
>>>         nbatch, npair = x.shape
>>>         x = x.reshape(-1,1)
>>>         x = self.fc2(self.fc1(x))
>>>         return x.reshape(nbatch, npair)

This Jastrow use two fully connected layers. The calculation of the first and second derivative are then done via automatic differentiation
as implemented in the `JastrowKernelElectronElectronBase` class.
To use this new Jastrow in the `SlaterJastrow` wave function ansatz we simply pass the class name as argument of the `jastrow_kernel` keyword argument :

>>> # define the wave function
>>> wf = SlaterJastrow(mol, kinetic='jacobi',
>>>                    backflow_kernel=MyBackflow
>>>                    configs='single_double(2,2)')

As previously, `mol` is an instance of the `qmctorch.scf.Molecule` class.