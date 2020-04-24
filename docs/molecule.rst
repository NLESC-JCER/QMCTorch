
Molecule in QMCTorch
--------------------------------
Molecules are defined via the `Molecule` class in QMC Torch. For example the following lines :

>>> mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69', unit='bohr'
>>>                calculator='adf', basis='dzp')

The `atom` keyword argument specifies the atom types and positions of the atoms in the molecule. 
Here we define a H2 molecule where both hydrogen atoms are on the z-axis located at +/- 0.6 bohr from the center. 
The units of the coordinates given for the `atom` keyword argument are specfied with the `unit` argument (bohr or angs).
It is also possible to read the coordinate from an xyz file using :

>>> mol = Molecule(atom='h2o.xyz', unit='bohr'
>>>                calculator='adf', basis='dzp')

where 'h2o.xyz' is the path to the xyz file. 

Interface with QM packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two other keyword arguments (``calculator`` and ``basis``) refers to the QM packages that can be used to define the basis set
used to describe the electronic strucrure of the molecule. The calculator refers to a particular QM package and so far only
``pyscf`` and ``adf`` are supported. ``basis``  refers to the basis set used for the calculation.
The value of the keyword argument should be a valid name of an **all electron** basis set.
