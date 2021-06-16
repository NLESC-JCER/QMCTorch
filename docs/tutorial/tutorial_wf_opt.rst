Wave function Optimization
====================================

We present here a complete example on how to use QMCTorch on a H2 molecule.
We first need to import all the relevant modules :

>>> from torch import optim
>>> from qmctorch.scf import Molecule
>>> from qmctorch.wavefunction import SlaterJastrow,
>>> from qmctorch.solver import SolverSlaterJastrow
>>> from qmctorch.sampler import Metropolis
>>> from qmctorch.utils import set_torch_double_precision
>>> from qmctorch.utils import (plot_energy, plot_data)

To obtain a bettter accuracy on our results we can switch to a double precision default
tensor type for pytorch :

>>> set_torch_double_precision()

The first step is to define a molecule. We here use a H2 molecule with both hydrgen atoms
on the z-axis and separated by 1.38 atomic unit. We also choose to use ADF as SCF calculator using
a double zeta + polarization (dzp) basis set.

>>> # define the molecule
>>> mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
>>>                calculator='adf',
>>>                basis='dzp',
>>>                unit='bohr')

We then define the QMCNet wave function relative to this molecule. We also specify here
the determinants we want to use in the CI expansion. We use here a to include all the single
and double excitation with 2 electrons and 2 orbitals

>>> # define the wave function
>>> wf = SlaterJastrow(mol, kinetic='jacobi',
>>>              configs='single_double(2,2)')

As a sampler we use a simple Metropolis Hasting with 1000 walkers. The walkers are initially localized around the atoms.
Each walker will perform 2000 steps of size 0.2 atomic unit and will only keep the last position of each walker (`ntherm=-1`).
During each move all the the electrons are moved simultaneously within a normal distribution centered around their current location.

>>> # define the sampler
>>> sampler = Metropolis(nwalkers=1000,
>>>                      nstep=2000, step_size=0.2,
>>>                      ntherm=-1, ndecor=100,
>>>                      nelec=wf.nelec, init=mol.domain('atomic'),
>>>                      move={'type': 'all-elec', 'proba': 'normal'})


We will use the ADAM optimizer implemented in pytorch with custom learning rate for each layer.
We also define a linear scheduler that will decrease the learning rate after 100 steps

>>> # optimizer
>>> lr_dict = [{'params': wf.jastrow.parameters(), 'lr': 1E-3},
>>>            {'params': wf.ao.parameters(), 'lr': 1E-6},
>>>            {'params': wf.mo.parameters(), 'lr': 1E-3},
>>>            {'params': wf.fc.parameters(), 'lr': 1E-3}]
>>> opt = optim.Adam(lr_dict, lr=1E-3)
>>>
>>> # scheduler
>>> scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.90)

We can now assemble the solver :

>>> # QMC solver
>>> solver = SolverSlaterJastrow(wf=wf, sampler=sampler, optimizer=opt, scheduler=None)

The solver needs to be configured. Many parameters of the optimization can be controlled as illustrated below

We can specify which observale to track during the optimization. Here only the local energies will be recorded
but one can also record the variational parameters
>>> solver.configure(track=['local_energy'])

Some variational parameters can be frozen and therefore not optimized. We here freeze the MO coefficients and the AO parameters
and therefore only the jastrow parametres and the CI coefficients will be optmized
>>> solver.configure(freeze=['ao', 'mo'])

Either the mean or the variance of local energies can be used as a loss function. We here use the mean
>> solver.configure(loss='energy')

The gradients can be computed directly via automatic differntiation or via a reduced noise formula
>>> solver.configure(grad='auto')

We also configure the resampling so that the positions of the walkers are updated by performing
25 MC steps from their previous positions after each optimization step.

>>> solver.configure(resampling={'mode': 'update',
>>>                             'resample_every': 1,
>>>                             'nstep_update': 25})




We can now run the optimization. We use here 250 optimization steps (epoch), using all the points
in a single mini-batch.

>>> data = solver.run(250)

Once the optimization is done we can plot the results

>>> plot_energy(solver.observable.local_energy, e0=-1.1645, show_variance=True)
>>> plot_data(solver.observable, obsname='jastrow.weight')

.. image:: ../../pics/h2_dzp.png