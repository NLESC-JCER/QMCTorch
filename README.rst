################################################################################
pyCHAMP
################################################################################

Quantum Monte Carlo code in Python

Introduction
*************

pyCHAMP allows to run small Variational QMC calculations in Python. Diffusion Monte Carlo is currently under developement Only a few features are currently supported : 

### Sampler : 
  * Metropolis-Hasting
  * Hamiltonian Monte-Carlo

### Optimizers :
  * Scipy Minimize routines (BFGS, Simplex, .... )
  * Linear Method
  
  
pyChamp tries to use `autograd` as much as possible to define the partial derivatives of the wave function, alleviating the necessaity to derive analytic expressions

