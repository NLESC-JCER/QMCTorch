---
title: 'QMCTorch: a PyTorch Implementation of Real-Space Quantum Monte Carlo Simulations of Molecular Systems'
tags:
  - Python
  - Deep Learning
  - Quantum Chemistry
  - Monte Carlo
  - Molecular Systems
authors:
  - name: Nicolas Renaud
    orcid: 0000-0001-9589-2694
    affiliation: 1
affiliations:
 - name: Netherlands eScience Center, Science Park 402, 1098 XH Amsterdam, The Netherlands
   index: 1
date: 11 April 2023
bibliography: paper.bib

---

# Summary

Quantum Monte-Carlo (QMC) simulations allow to compute the electronic structure of quantum systems
with high accuracy and can be parallelized over large compute resources. QMC relies on the variational principle and optimize a wave function ansatz to minimize the total energy of the quantum system. `QMCTorch` expresses this optimization process as a machine learning problem where the wave function ansatz is encoded
in a physically-motivated neural network. The use of `PyTorch` as a backend to perform the optimization, allows leveraging automatic differentiation and GPU computing to accelerate the development and deployment of QMC simulations. `QMCTorch` supports the use of both Gaussian and Slater type orbitals via interface to popular quantum chemistry packages `pyscf` and `ADF`.


# Statement of need

`QMCTorch` is a Python package using `PyTorch` [@pytorch] as a backend to perform Quantum Monte-Carlo (QMC) simulations, namely Variational Monte-Carlo,  of molecular systems. Many software such as `QMCPack`[@qmcpack], `QMC=Chem` [@qmcchem], `CHAMP` [@champ] provide high-quality implementation of advanced QMC methodologies in low-level languages (C++/Fortran).  Python implementations of QMC such as `PAUXY` [@pauxy] and `PyQMC` [@pyqmc] have also been proposed to facilitate the use and development of QMC techniques. Large efforts have been made to leverage recent development of deep learning techniques for QMC simulations with for example the creation of neural-network based wave-function ansatz [@paulinet; @ferminet] that have lead to very interesting results. `QMCTorch` allows to perform QMC simulations using physically motivated neural network architectures that closely follow the wave function ansatz used by QMC practitioners. Its architecture allows to rapidly explore new functional forms of some key elements of the wave function ansatz. Users do not need to derive analytical expressions for the gradients of the total energy w.r.t. the variational parameters, that are simply obtained via automatic diffentiation. This includes for example the parameters of the atomic orbitals that can be varioationally optimized and the atomic coordinates that allows `QMCTorch` to perform geometry optimization of molecular structures. In addition, the GPU capabilities offered by `PyTorch` combined with the parallelization over multiple computing nodes obtained via `Horovod` [@horovod], allow to deploy the simulations on large heterogenous computing architectures. In summary, `QMCTorch` provides QMC practionners a framework to rapidly protoytpe new ideas and to test them using modern computing ressources.


# Wave Function Ansatz
![General architecture of the neural network used by `QMCTorch` to encode the wave function ansatz. The neural network computes and assembles the different elements of the wave function ansatz and can be used to compute the electronic density required for the sampling and the total energy of the system required for the wave function optimization.\label{fig:arch}](qmctorch2.png)

The neural network used to encode the wave-function ansatz used in `QMCTorch` is shown in Fig. \ref{fig:arch}. As common in QMC simulations, the wave function is given by the product of a Jastrow factor, $J(r)$, that accounts for electronic correlations and a sum of Slater determinants, $D^\updownarrow(r_\updownarrow)$, built over the molecular orbitals of the spin up and down electrons:  $\Psi(r) = J(r)\sum_n c_n D^\uparrow(r_\uparrow)D^\downarrow(r_\downarrow)$.

**Jastrow Factor** The `Jastrow` layer computes the sum of three components: an electron-electron term $K_{ee}$; an electron-nuclei term $K_{en}$; and a three body electron-electron-nuclei term $K_{een}$. The sum is then exponentiated to give the Jastrow factor: $J(r_{ee}, r_{en}) = \exp\left( K_{ee}(r_{ee})+K_{en}(r_{en}) + K_{een}(r_{ee},r_{en})\right)$ where $r_{ee}$ and $r_{en}$ are the electron-electron and electron-nuclei distances. Several well-known Jastrow factor functional forms, as for example the electron-electron Pade-Jastrow: $K(r_{ee}) = \frac{\omega_0 r_{ee}}{1 + \omega r_{ee}}$, where $\omega$ is a variational parameter, are already implemented and available for use. Users can also define their own functional forms for the different kernel functions, $K$, and explore their effects on the resulting optimization.  

**Backflow Transformation** The backflow transformation layer, `BF`, creates quasi-particles by mixing the electronic positions of the electrons: $q_i = r_i + \sum_{i\neq j} K_{BF}(r_{ij})(r_i-r_j)$ [@backflow]. Well-known transformations such as: $K_{BF} = \frac{\omega}{r_{ij}}$ where $\omega$ is a variational parameter, are already implemented and ready to use. Users can also easily specify the kernel of the backflow transformation, $K_{BF}$ to explore its impact on the wave function optimization.

**Atomic Orbitals** The Atomic Orbital layer `AO` computes the values of the different atomic orbitals of the system at all the positions $q_e$. Both Slater type orbitals (STOs) and Gaussian type orbitals (GTOs) are supported. The initial parameters of the AOs are extracted from popular quantum chemistry codes, `pyscf` [@pyscf] and `ADF` [@adf].  During the optimization, the parameters of the AOs (exponents, coefficients) are variational parameters that can be optimized to minimize the total energy. Since GTOs can introduce a significant amount of noise in the QMC simulations, `QMCTorch` offers the possibility to fit GTOs to single exponent STOs.

**Molecular Orbitals** The Molecular Orbital layer, `MO`, computes the values of all the MOs at the positions of the quasi particles. The MO layer is a simple linear transformation defined by $\textnormal{MO} =  \textnormal{AO} \times W^T_{SCF}$, where $W^T_{SCF}$ is the matrix of the MOs coefficients on the AOs. The initial values of these coefficients are obtained from a Hartree-Fock (HF) or Density Functional Theory (DFT) calculation of the system via `pyscf` or `ADF`. These coefficients are then variational parameters that can be optimized to minimize the total energy of the system. 

**Slater Determinants** The Slater determinants layer, `SD`, extracts the spin up/down  matrices of the different electronic configurations specified by the user. Users can freely define the number of electrons as well as the number and types of excitations they want to include in the definition of their wave function ansatz. The `SD` layer will extract the corresponding matrices, multiply their determinants and sum all the terms. The `CI` coefficients of the sum can be freely initialized and optimized to minimize the total energy.

The Jastrow factor and the sum of Slater determinants are then multiplied to yield the final value of the wave function calculated for the electronic and atomic positions $\Psi(R)$ with $R = \{r_e, R_{at}\}$. Note that the backflow transformation and Jastrow factor are optional and can be individually removed from the definition of the wave function ansatz. 

# Sampling, Cost Function & Optimization 

QMC simulations use samples of the electronic density to approximate the total energy of the system. In `QMCTorch`, Markov-Chain Monte-Carlo (MCMC) techniques, namely Metropolis-Hasting and Hamiltonian Monte-Carlo, are used to obtained those sample. Each sample, $R_i$, contains the positions of all the electrons contained in the system. MCMC techniques require the calculation of the density for a given positions of the electrons: $\rho(R_i) = |\Psi(R_i)|^2$ that can simply obtained by squaring the result of a forward pass of the network described above.

The value of local energy of the system is then computed at each sampling point and these values are summed up to compute the total energy of the system: $E = \sum_i \frac{H\Psi(R_i)}{\Psi(R_i)}$, where $H$ is the Hamiltonian of the molecular system: $H = -\frac{1}{2}\sum_i \Delta_i + V_{ee} + V_{en}$, with $\Delta_i$ the Laplacian w.r.t the i-th electron, $V_{ee}$ the coulomb potential between the electrons and $V_{en}$ the electron-nuclei potential. in `QMCTorch`, the calculation of the Laplacian of the Slater determinants can be performed using automatic differentiation but analytical expressions have also been implemented as they are computationally more robust and less expensive [@jacobi_trace]. The gradients of the total energy w.r.t the variational parameters of the wave function, i.e. $\frac{\partial E}{\partial \theta_i}$ are simply obtained via automatic differentiation. Thanks to this automatic differentiation, users can define new kernels for the backflow transformation and Jastrow factor without having to derive analytical expressions of the energy gradients. 

Any optimizer included in `PyTorch` (or compatible with it) can then used to optimize the wave function. This gives users access to a wide range of optimization techniques that they can freely explore for their own use cases. Users can also decide to freeze certain variational parameters or defined different learning rates for different layers. Note that the positions of atoms are also variational parameters, and therefore one can perform geometry optimization using `QMCTorch`. At the end of the optimization, all the information relative to the simulations are dumped in a dedicated HDF5 file to enhance reproducibility of the simulations.

# Example
```python
from torch import optim
from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow
from qmctorch.solver import Solver
from qmctorch.sampler import Metropolis

# define the molecule
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69', calculator='pyscf', basis='sto-3g')

# define the wave function ansatz
wf = SlaterJastrow(mol, configs='single_double(2,2)').gto2sto()

# define a Metroplos Hasting Sampler
sampler = Metropolis(nwalkers=5000, nstep=200, nelec=wf.nelec, init=mol.domain('atomic'))

# define the optimizer
opt = optim.Adam(wf.parmaters(), lr=1E-3)

# define the solver
solver = Solver(wf=wf, sampler=sampler, optimizer=opt)

# optimize the wave function
obs = solver.run(50)
```

![Result of the  optimization of the wave function of LiH and NH3 using atomic atomic orbitals provided by `pyscf`, `ADF` and also a STO fit of the `pyscf` atomic orbitals. The vertical axis shows the difference between the variational energy and the true groud state energy. The horizontal dashed line indicate the Hatree-Fock energy. \label{fig:results}](qmctorch_results.png)

The snippet of code above shows a typical example of `QMCTorch` script. A `Molecule` object is first created by specifying the atomic positions and the calculator required to run the HF or DFT calculations (here `pyscf` using  a `sto-3g` basis set). This molecule is then used to create a `SlaterJastrow` wave function ansatz. Other options, such as the required Jastrow kernel, active space, and the use of GPUs can also be specified here. A sampler and optimizer are then defined that are then used with the wave function to instantiate the solver. This solver can then be used to optimize the variational parameters, that is done here through 50 epochs. 

Fig. \ref{fig:results} shows typical optimization runs for two different molecular structures, LiH and NH3 using atomic orbitals provided by `pyscf`, `ADF` and also a STO fit of the `pyscf` atomic orbitals. As seen in this figure, the variance of the local energy values obtained with the GTOs provided by `pyscf` is a limiting factor for the optimization. A simple STO fit of these atomic orbitals leads to variance comparable to those obtained with the STOs of `ADF`.

# Acknowledgements

We acknowledge contributions from Felipe Zapata, Matthijs de Witt and guidance from Claudia Filippi. The development of the code was done during the project "A Light in the Dark: quantum Monte-Carlo meets energy conversion" from the Joint CSER and eScience program for Energy Research (JCER2017) funded by the Nederlandse Organisatie voor Wetenschappelijk Onderzoek (NWO) and the Netherlands eScience Center, project number CSER.JCER.022

# References
