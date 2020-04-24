QMCNet : A Neural Network for QMC 
-----------------------------------

QMCTorch proposes to leverage the optimization capabilities of PyTorch to optimize QMC wave function. 
To that end, the wave function ansatz has been incorporated in the dedicated neural network architecture presented below and dubbed QMCNet:

.. image:: ../pics/mol_nn.png

As seen on this figure, the input of the network are the positions of the walkers that should be sampled from the density of the wavefunction.
The first layer of the QMCNet computes the values of all the atomic orbitals at the position of each electron contained in a given walker configuration. 
This new layer has been implemented in Pytorch and allows the optimizaiton of the atomic position (geometry optimization), and of the atomic basis parameters (basis exponents and coefficients).
The second layer transforms the atomic orbitals in molecular orbitals using a linear map whose weights are the molecular orbitals coefficients. The third layer, dubbed Slater Pooling,
computes the values of all the desired Slater determinants from the values of the molecular orbilals. The fina layer sums up the slater determinant using the CI coefficients as weight

In parallel a Jastrow layer has been implemented in Pytorch and allows the calculation of the Jastrow factor directly from the walkers positions. 
The Jastrow factor is multiplied with the sum of the slater determinant to obtain the value of the wave function. 

The QMCnet is implemented in the ``Orbital`` :

>>> wf = Orbital(mol, configs='cas(2,2)')

The ``Orbital`` takes as first mandiatory argument a ``Molecule`` instance. The Slater determinants required in the calculation
are specified with the ``configs`` arguments which can take the following values :

  * ``configs='ground_state'`` : only the ground state SD 
  * ``configs='cas(n,m)'`` : complete active space using n electron and m orbitals
  * ``configs='single(n,m)'`` : only single excitation using n electron and m orbitals
  * ``configs='single_double(n,m)'`` : only single/double excitation using n electron and m orbitals

The ``Orbital`` class accepts other initialisation arguments to fine tune some advanced settings. The default values
of these arguments are adequeate for most cases.