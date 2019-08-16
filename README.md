# DeepQMC

Deep Learning for Quantum Monte Carlo Simulations

## Introduction

DeepQMC allows to leverage deep learning to optimize QMC wave functions. The package offers solutions to optimize particle in a box model and molecular systems. It relies heavily on `pytorch` for the deep learning part and on `pyscf` to obtain the first guess of the molecular orbitals. The three manin ingredients of any calculations are :

  * a neural network calculation the value of the wave function at a given point
  * a sampler able to generate sampling points of the wave function
  * an optimizer (as those provided by `pytorch`) 

Several MC samplers are implemented :

  * Metropolis-Hasting
  * Hamiltonian Monte-Carlo

and more will be added. Beyond facilitating the optimization of the wave fuction parameters, `autograd` is also leveraged for example to apply the kinetic operator on the wave function.


## Harmonic Oscillator in 1D

We illustrate here how to optimize a simple harmonic oscillator in 1D using DeepQMC. The `pot_func` function defines the potential that is here a simple harmonic oscillator. The `sol_func` function gives the analytical solution of the problem and is only use for plotting purposes.



```python
import torch
import torch.optim as optim

from deepqmc.wavefunction.wf_potential import Potential
from deepqmc.sampler.metropolis import  Metropolis
from deepqmc.solver.deepqmc import DeepQMC
from deepqmc.solver.plot import plot_results_1d, plotter1d

def pot_func(pos):
    '''Potential function desired.'''
    return  0.5*pos**2

def sol_func(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    return torch.exp(-0.5*pos**2)

# box
domain, ncenter = {'xmin':-5.,'xmax':5.}, 5

# wavefunction
wf = Potential(pot_func,domain,ncenter,nelec=1)

#sampler
sampler = Metropolis(nwalkers=250, nstep=1000, 
                     step_size = 1., nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-5,'max':5})

# optimizer
opt = optim.Adam(wf.parameters(),lr=0.01)

# define solver
qmc = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)

# train the wave function
pos,obs_dict = qmc.train(100, loss = 'variance',
                         plot = plotter1d(wf,domain,50,sol=sol_func) )

# plot the final wave function 
plot_results_1d(qmc,obs_dict,domain,50,sol_func,e0=0.5)
```


After defining the domain in `domain` and the number of basis function in `ncenter`, we instantialte the `Potential` wave function class. This class defines a very simple neural network that, given a position computes the value of the wave function at that point. This neural network is composed of a layer of radial basis functions followed by a fully conneted layer to sum them up:

<p align="center">
<img src="./pics/rbf_nn.png" title="RBF neural network">
</p>

The then instantiate the sampler, here a simple `Metroplis` scheme. The sampler is used to sample the wave function and hence generate a bach of sampling points. These points are used as input of the neural network the compute the values of wave function at those points. We finally select the `Adam` optimizer to optimize the wave function paramters. 

We then define a `DeepQMC` instance that ties all the elements together and train th model to optimize the wave function paramters. We here use the variance of the sampling point energy as a loss and run 100 epochs. Many more parameters are accessible in the training routines.

After the optimization, the following result is obtained:

<p align="center">
<img src="./pics/ho1d.png" title="Results of the optimization">
</p>


## Dihydrogen molecule

We use a similar network architecture to optimize QMC wavefunction for molecular systems. This architecture is depicted below. 

<p align="center">
<img src="./pics/mol_nn.png" title="Neural network used for molecular systems">
</p>

Starting from the positions of the electrons in the system, we have define a network layer that evaluate the values of all the atomic orbitals at all the electron positions. This layer has several variational paramters (atomic positions, basis function exponents and coefficients) that can be optimized during the training. 

From the atomic orbital values we compute the values of the molecular orbitals, there again for the positions of all the electrons. This achieved by a simple linear layer whose transformation matrix is given by the molecular orbital coefficients. These coefficients are variational parameters of the layer and can therefore also be optimized.

We then have defined a Slater pooling layer that that computes values of all the required Slater determinant. Finally a fully connected layer sums up the determinants. The weight of this last layer are the CI coefficients that can as well be optimized.

In parallel we also have defined a Jastrow layer that computes the e-e distance and the value of the Jastrow factor. There again the parameters of the layer can be  optimized during the training of the wave function.
