Creating your own Jastrow Factor
====================================

We present here how to create your own electron-electron Jastrow factor and use it in QMCTorch

During the import you must import the base class of the electron-electron Jastrow.

>>> from qmctorch.scf import Molecule
>>> from qmctorch.wavefunction import SlaterJastrow,
>>> from qmctorch.wavefunction.jastrows.elec_elec.kernels import JastrowKernelElectronElectronBase


We can then use this base class to create a new Jastrow Factor. This is done in the same way one would create
a new neural network layer in pytorch.

>>> class MyJastrow(JastrowKernelElectronElectronBase)
>>>
>>>     def __init__(self, nup, ndown, cuda, size=16):
>>>         super().__init__(nup, ndown, cuda)
>>>         self.fc1 = nn.Linear(1, size, bias=False)
>>>         self.fc2 = nn.Linear(size, 1, bias=False)
>>>
>>>     def forward(self, x):
>>>         nbatch, npair = x.shape
>>>         x = x.reshape(-1,1)
>>>         x = self.fc2(self.fc1(x))
>>>         return x.reshape(nbatch, npair)

As seen above the prototype of the class constructor must be:

>>> def __init__(self, nup, ndown, cuda, **kwargs)

The list of keyword argument can contain any pairs such as ``size=16``.

This Jastrow use two fully connected layers. The size of the hidden layer is here controlled by a keyword argument ``size`` whose defauilt value is 16
It is important to note that the calculation of the first and second derivative of the jastrow kernel wrt the electronic positions are then done via automatic differentiation
as implemented in the `JastrowKernelElectronElectronBase` class. Hence there is no need to derive and implement these derivatives. However it
is necessary that the ``forward`` function, which takes as input a ``torch.tensor`` of
dimension ``[Nbatch, Npair]`` first reshape this tensor to ``[Nbatch*Npair,1]``, then applies the transformation on this tensor and finally reshape
the output tensor to ``[Nbatch, Npair]``.

To use this new Jastrow in the `SlaterJastrow` wave function ansatz we simply pass the class name as argument of the `jastrow_kernel` keyword argument. It is also
possible to specify the values of the keyword argument ``size`` with the ``jastrow_kernel_kwargs``. As seen below the pair of keyword argument and its value is passed as
a python dictionary :

>>> # define the wave function
>>> wf = SlaterJastrow(mol, kinetic='jacobi',
>>>                    jastrow_kernel=MyJastrow
>>>                    jastrow_kernel_kwargs={'size' : 64}
>>>                    configs='single_double(2,2)')

As previously, `mol` is an instance of the `qmctorch.scf.Molecule` class.