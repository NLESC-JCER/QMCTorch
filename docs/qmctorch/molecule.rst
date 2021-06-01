
Creating a molecule
========================================

In this tutorial we present how to create a molecule and run the SCF calculation. First, the `Molecule` class must be imported :

>>> from qmctorch.scf import Molecule

This class can interface with `pyscf` and `ADF` to perform SCF calculations. Of course both software use different types of
atomic orbitals, respectively Gaussian type orbitals for `pyscf` and Slater type orbitals for `ADF`.


Geometry of the molecule
------------------------------------------

The geometry of the molecule can be specified through the `atom` keyword of the `Molecule` class. The units of the positions, `bohr` or `angs` (default is 'bohr')
can also be specified via the `unit` keyword argument.  The geometry can be passed as a single string

>>> Molecule(atom = 'H 0. 0. 0; H 0. 0. 1.', unit='bohr')

or via an XYZ file containing the geomtry of the molecular structure

>>> Molecule(atom='h2.xyz', unit='angs')

SCF calculations
--------------------------------------------

As mentionned above `QMCTorch` can use `pyscf` or `ADF` to perform SCF calculation on the molecular structure. At the moment only Hartree-Fock calculations
are supported but DFT calculations will be implemented later. We present here how to perform these SCF calculations


Gaussian orbitals with pyscf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use `pyscf` to compute the molecular orbitals of the system the `calculator` simply need to be set to `pyscf` as shown below.

>>> # define the molecule
>>> mol = Molecule(atom='H 0. 0. 0; H 0. 0. 1.', unit='angs',
>>>                calculator='pyscf', basis='sto-3g')

The `basis` keyword specify which basis set to use in the calculation. We use here a small `STO-3G` basis set. The exhaustive list of supported basis
set can be found here : https://pyscf.org/user/gto.html


Slater orbitals with ADF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If a valid SCM license is found  QMCTorch can use `ADF` as a backend by simply switching the `calculator` to 'adf' as below :

>>> # define the molecule
>>> mol = Molecule(atom='H 0. 0. 0; H 0. 0. 1.', unit='angs',
>>>               calculator='adf', basis='dzp')

Here as well the ``basis`` keyword argument specifies the basis set used in the scf calculation.
The list of supported basis set can be found here : https://www.scm.com/doc/ADF/Input/Basis_sets_and_atomic_fragments.html

Additional basis sets, namely VB1, VB2, VB3, CVB1, CVB2 and CVB3, are available. These are STO valence and core-valence basis set presented by Ema et. al
in  "Polarized basis sets for Slater-type orbitals: H to Ne atoms", https://doi.org/10.1002/jcc.10227. Changing the ``basis``
 keyword argument to : ``basis=VB1``` will for examle use the small VB1 basis set during the SCF calculation.

Loading a SCF calculation
----------------------------------

By default QMCTorch will create a HDF5 file containing all the required information about the molecule and SCF calculation. The name of
this file is given by the name of the molecule, the calculator name and the basis set, e.g. `LiH_adf_dzp.hdf5` or 'water_pyscf_sto3g.xyz'. This files
can be loaded to instanciate the molecule object through the `load` keyword argument:

>>> mol = Molecule(load='LiH_adf_dzp.hdf5')

