from qmctorch.scf import Molecule

# Select the SCF calculator
calc = ['pyscf',            # pyscf  
        'adf',          # adf 2019 
        'adf2019'               # adf 2020+
    ][1]

# select an appropriate basis
basis = {
    'pyscf'  : 'sto-6g',
    'adf'    : 'VB1',
    'adf2019': 'dz'
}[calc]

# do the scf calculation
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
            calculator=calc,
            basis=basis,
            unit='bohr')



