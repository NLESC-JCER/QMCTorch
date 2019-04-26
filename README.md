# pyCHAMP

Quantum Monte Carlo code in Python

## Introduction


pyCHAMP allows to run small Variational QMC calculations in Python. Diffusion Monte Carlo is currently under developement Only a few features are currently supported : 

### Sampler : 

  * Metropolis-Hasting
  * Hamiltonian Monte-Carlo

### Optimizers :

  * Scipy Minimize routines (BFGS, Simplex, .... )
  * Linear Method
  
  
`pyChamp` tries to use `autograd` as much as possible to define the partial derivatives of the wave function, alleviating the necessaity to derive analytic expressions

## Example : VMC Single point calculation 1D Harmonic oscillator with 1-electron

```python

from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.solver.vmc import VMC

class HarmOsc1D(WF):

    def __init__(self,nelec,ndim):
        WF.__init__(self, nelec, ndim)

    def values(self,parameters,pos):
        ''' Compute the value of the wave function.
        Args:
            parameters : parameters of th wf
            x: position of the electron
        Returns: values of psi
        '''
    
        beta = parameters[0]
        return np.exp(-beta*pos**2).reshape(-1,1)

    def nuclear_potential(self,pos):
        return 0.5*pos**2 

    def electronic_potential(self,pos):
        return 0
        
wf = HarmOsc1D(nelec=1, ndim=1)
sampler = METROPOLIS(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=1, domain = {'min':-2,'max':2})

vmc = VMC(wf=wf, sampler=sampler, optimizer=None)
opt_param = [0.5]
pos, e, s = vmc.single_point(opt_param)

print('Energy   : ', e)
print('Variance : ', s)
vmc.plot_density(pos)
```


This script will output :



Example : 3D Harmonic oscillator with 1-electron
*****************************************************
