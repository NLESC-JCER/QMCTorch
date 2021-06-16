Creating your own Backflow transformation
==============================================

We present here how to create your own backflow transformation. During the import you must import the base class of the backflow kernel

>>> from qmctorch.scf import Molecule
>>> from qmctorch.wavefunction import SlaterJastrowBackFlow
>>> from qmctorch.wavefunction.orbitals.backflow.kernels import BackFlowKernelBase


We can then use this base class to create a new backflow transformation kernel.
This is done in the same way one would create a new neural network layer in pytorch

>>> class MyBackflow(BackFlowKernelBase):
>>>
>>>     def __init__(self, mol, cuda, size=16):
>>>         super().__init__(mol, cuda)
>>>         self.fc1 = nn.Linear(1, size, bias=False)
>>>         self.fc2 = nn.Linear(size, 1, bias=False)
>>>
>>>     def forward(self, x):
>>>         original_shape = x.shape
>>>         x = x.reshape(-1,1)
>>>         x = self.fc2(self.fc1(x))
>>>         return x.reshape(*original_shape)

This backflow transformation consists of two fully connected layers. The calculation of the first and second derivative are then done via automatic differentiation
as implemented in the `BackFlowKernelBase` class. To use this new kernel in the `SlaterJastrowBackFlow` wave function ansatz we simply pass the class name as argument of the `backflow_kernel` keyword argument :

>>> # define the wave function
>>> wf = SlaterJastrowBackFlow(mol, kinetic='jacobi',
>>>                    backflow_kernel=MyBackflow,
>>>                    backflow_kernel_kwargs={'size' : 64},
>>>                    configs='single_double(2,2)')
