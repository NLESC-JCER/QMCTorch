Slater Jastrow Wave Function
-----------------------------------


The ``SlaterJastrow`` neural-network wavefunction ansatz matches closely the wave function ansatz commonly used in QMC simulations. The wave function
is here expressed as

.. math::

    \Psi(R) = J(R) \sum_n c_n D_n^{\uparrow} D_n^{\downarrow}

The term `J(R)` is the so called Jastrow factor that captures the electronic correlation. The Jastrow factor is given by :

.. math::

    J(R) = \exp\left(  \sum_{i<j} \text{Kernel}(r_{ij}) \right)

where the sum runs over all the electron pairs and where the kernel function defines the action of the Jastrow factor. A common expression for the
kernel (and the default option for QMCTorch) is the Pade-Jastrow form given by:

.. math::

    \text{Kernel}(r_{ij}) = \frac{\omega_0 r_{ij}}{1+\omega r_{ij}}

where :math: `\omega_0` is a fixed coefficient equals to 0.25(0.5) for antiparallel(parallel) electron spins and :math: `\omega` a variational parameter.

The determinantal parts in the expression of :math: `\Psi` are given by the spin-up and spin-down slater determinants e.g. :

.. math::

    D_n^{\uparrow} = \frac{1}{\sqrt{N}} \begin{vmatrix} & & \\ & \phi_j(r_i) & \\ & & \end{vmatrix}


Implementation
^^^^^^^^^^^^^^^^^^^^^^^^

The Slater Jastrow function is implemented in QMCTorch as represented in the figure below :

.. image:: ../../pics/mol_nn.png

As seen on this figure, the input of the network are the positions of the walkers that should be sampled from the density of the wavefunction.
The first layer of the network computes the values of all the atomic orbitals at the position of each electron contained in a given walker configuration.
This new layer has been implemented in Pytorch and allows the optimizaiton of the atomic position (geometry optimization), and of the atomic basis parameters (basis exponents and coefficients).
The second layer transforms the atomic orbitals in molecular orbitals using a linear map whose weights are the molecular orbitals coefficients. The third layer, dubbed Slater Pooling,
computes the values of all the desired Slater determinants from the values of the molecular orbilals. The fina layer sums up the slater determinant using the CI coefficients as weight

In parallel a Jastrow layer has been implemented in Pytorch and allows the calculation of the Jastrow factor directly from the walkers positions.
The Jastrow factor is multiplied with the sum of the slater determinant to obtain the value of the wave function.

Usage
^^^^^^^^^^^^^^^^^^^^^^^
The ``SlaterJastrow`` wave function can instantiated following :

>>> wf = SlaterJastrow(mol, configs='single_double(2,2)', jastrow_kernel=PadeJastrowKernel)

The ``SlaterJastrow`` takes as first mandiatory argument a ``Molecule`` instance. The Slater determinants required in the calculation
are specified with the ``configs`` arguments which can take the following values :

  * ``configs='ground_state'`` : only the ground state SD
  * ``configs='cas(n,m)'`` : complete active space using n electron and m orbitals
  * ``configs='single(n,m)'`` : only single excitation using n electron and m orbitals
  * ``configs='single_double(n,m)'`` : only single/double excitation using n electron and m orbitals

Finally the kernel function of the Jastrow factor can be specifed using the ``jastrow_kernel``
The ``SlaterJastrow`` class accepts other initialisation arguments to fine tune some advanced settings. The default values
of these arguments are adequeate for most cases.

Orbital dependent Jastrow factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Jastrow factor can be made orbital dependent with the ``SlaterOrbitalDependentJastrow``

>>> from qmctorch.wavefunction import SlaterOrbitalDependentJastrow
>>> from qmctorch.wavefunction.jastrows.elec_elec.kernels import PadeJastrowKernel
>>> wf = SlaterOrbitalDependentJastrow(mol, configs='single_double(2,4)'
>>>                                    jastrow_kernel=PadeJastrowKernel)
