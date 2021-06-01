Slater Jastrow Backflow Wave Function
----------------------------------------

The Slater Jastrow Backflow wave function builds on the the Slater Jastrow wavefunction but adds a backflow transformation to
the electronic positions. Following this transformation, each electron becomes a quasi-particle whose position depends on all
electronic positions. The backflow transformation is given by :

.. math::

    q(x_i) = x_i + \sum_{j\neq i} \text{Kernel}(r_{ij}) (x_i-x_j)

The kernel of the transformation can be any function that depends on the distance between two electrons. A popular kernel
is simply the inverse function :

.. math::
    \text{Kernel}(r_{ij}) = \frac{\omega}{r_{ij}}

and is the default value in QMCTorch. However any other kernel function can be implemented and used in the code.

The wave function is then constructed as :

.. math::

    \Psi(R) = J(R) \sum_n c_n D_n^{\uparrow}(Q) D_n^{\downarrow}(Q)

The Jastrow factor is still computed using the original positions of the electrons while the determinant part uses the
backflow transformed positions.

Orbital Dependent Backflow Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The backflow transformation can be different for each atomic orbitals.

.. math::

    q^\alpha(x_i) = x_i + \sum_{j\neq i} \text{Kernel}^\alpha(r_{ij}) (x_i-x_j)

where each orbital has its dedicated backflow kernel. This provides much more flexibility when optimizing the wave function.

Usage
^^^^^^^^^^^^^^^^^^^^^

This wave function can be used with

>>> from qmctorch.wavefunction import SlaterJastrowBackFlow
>>> from qmctorch.wavefunction.orbitals.backflow.kernels import BackFlowKernelInverse
>>> from qmctorch.wavefunction.jastrows.elec_elec.kernels.pade_jastrow_kernel import PadeJastrowKernel
>>>
>>> wf = SlaterJastrowBackFlow(self.mol, kinetic='jacobi',
>>>                            configs='single_double(2,2)',
>>>                            jastrow_kernel=PadeJastrowKernel,
>>>                            orbital_dependent_backflow=False,
>>>                            backflow_kernel=BackFlowKernelInverse)

Compared to the ``SlaterJastrow`` wave function, the kernel of the backflow transformation must be specified.
By default the inverse kernel will be used. Orbital dependent backflow orbitals can be easily achieved by using ``orbital_dependent_backflow=True``