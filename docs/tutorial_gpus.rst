GPUs and multi-GPUs Support
==============================================

.. warning::
    The use of GPU and mutli-GPU is under developpement and hasn't been
    thoroughly tested yet. Proceed with caution !


Using pytorch as a backend, QMCTorch can leverage GPU cards available on your hardware.
You of course must have the CUDA version of pytorch installed (https://pytorch.org/)


Running on a single GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The use of GPU acceleration has been streamlined in QMCTorch, the only modification
you need to do on your code is to specify `cuda=True` in the declaration of the wave function :


>>> # define the wave function
>>> wf = SlaterJastrow(mol, kinetic='jacobi',
>>>             configs='cas(2,2)',
>>>             use_jastrow=True,
>>>             cuda=True)

This will automatically port all the necesaary tensors to the GPU and offload all the corresponding operation
there.

Multi-GPU support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The use of multiple GPUs is made possible through the `Horovod` library : https://github.com/horovod/horovod
A dedicated QMCTorch Solver has been developped to handle multiple GPU. To use this solver simply import it
and use is as the normal solver and only a few modifications are required to use horovod :


>>> import horovod.torch as hvd
>>> from deepqmc.solver.solver_slater_jastrow_horovod import SolverSlaterJastrowHorovod
>>>
>>> hvd.init()
>>> if torch.cuda.is_available():
>>>    torch.cuda.set_device(hvd.local_rank())
>>>
>>> # define the molecule
>>> mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
>>>                calculator='pyscf', basis='sto-3g',
>>>               unit='bohr', rank=hvd.local_rank())
>>>
>>> ...
>>>
>>> solver = SolverSlaterJastrowHorovod(wf=wf, sampler=sampler,
>>>                              optimizer=opt, scheduler=scheduler,
>>>                              rank=hvd.rank())
>>> ....

As you can see some classes need the rank of the process when they are defined. This is simply
to insure that only the master process generates the HDF5 files containing the information relative to the calculation.

The code can then be launched using the `horovodrun` executalbe :

::

    horovodrun -np 2 python <example>.py

See the horovod documentation for more details : https://github.com/horovod/horovod


This solver distribute the `Nw` walkers over the `Np` process . For example specifying 2000 walkers
and using 4 process will lead to each process using only 500 walkers. During the optimizaiton of the wavefunction
each process will compute the gradients of the variational parameter using their local 500 walkers.
The gradients are then averaged over all the processes before the optimization step takes place. This data parallel
model has been greatly succesfull in machine learning applications (http://jmlr.org/papers/volume20/18-789/18-789.pdf)