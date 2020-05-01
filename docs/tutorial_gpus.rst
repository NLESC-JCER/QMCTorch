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
>>> wf = Orbital(mol, kinetic='jacobi',
>>>             configs='cas(2,2)',
>>>             use_jastrow=True,
>>>             cuda=True)

This will automatically port all the necesaary tensors to the GPU and offload all the corresponding operation
there.

Multi-GPU support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The use of multiple GPUs is made possible through the `Horovod` library : https://github.com/horovod/horovod
A dedicated QMCTorch Solver has been developped to handle multiple GPU. To use this solver simply import it 
and use is as the normal solver. You simply need to give each card it's own horovod rank.


>>> import horovod.torch as hvd
>>> from deepqmc.solver.solver_orbital_horovod import SolverOrbital
>>>
>>> hvd.init()
>>> if torch.cuda.is_available():
>>>    torch.cuda.set_device(hvd.local_rank())
>>>
>>> .... 
>>>
>>> solver = SolverOrbital(wf=wf, sampler=sampler, optimizer=opt)
>>>

The code can then be launched using the `horovodrun` executalbe :

::

    horovodrun -np 4 - H localhost:4 python <example>.py

See the horovod documentation for more details : https://github.com/horovod/horovod