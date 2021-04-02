Wave function Optimization
====================================

We present here a complete example on how to use QMCTorch on a H2 molecule.
We first need to import all the relevant modules :

>>> from torch import optim
>>> from qmctorch.wavefunction import SlaterJastrow, Molecule
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
the determinants we want to use in the CI expansion. We use here a CAS(2,2).

>>> # define the wave function
>>> wf = SlaterJastrow(mol, kinetic='jacobi',
>>>              configs='cas(2,2)',
>>>              use_jastrow=True)

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

The solver needs to be configured. We set the task to wave function optimization and freeze here the
variational parameters of the atomic orbitals and molecular orbitals. Hence only the jastrow factor and the CI coefficients
will be optimized.

>>> # optimize the wave function
>>> solver.configure(task='wf_opt', freeze=['ao', 'mo'])
>>> solver.track_observable(['local_energy'])

We also configure the resampling so that the positions of the walkers are updated by performing
25 MC steps from their previous positions after each optimization step.

>>> solver.configure_resampling(mode='update', resample_every=1, nstep_update=25)

We can now run the optimization. We use here 250 optimization steps (epoch), using all the points
in a single mini-batch. The energy of the system is used as a training loss and the gradients of the variational
parameters are computed using a low-variance expression (`grad='manual'`).

>>> data = solver.run(5, batchsize=None, loss='energy', grad='manual', clip_loss=False)

Once the optimization is done we can plot the results

>>> plot_energy(solver.observable.local_energy, e0=-1.1645, show_variance=True)
>>> plot_data(solver.observable, obsname='jastrow.weight')

.. image:: ../pics/h2_dzp.png