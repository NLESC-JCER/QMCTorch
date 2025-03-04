from pyscf import gto, scf, cc
import numpy as np
import matplotlib.pyplot as plt 

mol = gto.M(atom="H 0 0 0; H 0 0 0.74")
cc_scanner = cc.CCSD(scf.RHF(mol)).nuc_grad_method().as_scanner()

dist = np.linspace(0.25,1.5,15)
energies = []
for d in dist:
    atom = 'H 0 0 0; H 0 0 %f' %d
    e,g = cc_scanner(gto.M(atom=atom))
    energies.append(e)    

plt.plot(dist, energies)
plt.show()