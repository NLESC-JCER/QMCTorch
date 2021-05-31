Geometry Optimization
====================================

We present here a complete example on how to use QMCTorch on a H2O molecule.
As previously the firs task is to import all the modules needed

>>> from torch import optim
>>> from torch.optim import Adam
>>>  from qmctorch.wavefunction import SlaterJastrow
>>> from qmctorch.solver import SolverSlaterJastrow
>>> from qmctorch.samplerimport Metropolis
>>> from qmctorch.scf import Molecule
>>> from qmctorch.utils import plot_energy

We then define the molecule. We read here an xyz file of a water molecule
where the three atoms are on the same line.

>>> # define the molecule
>>> mol = Molecule(atom='water_line.xyz', unit='angs',
>>>                calculator='pyscf', basis='sto-3g')


The QMCNet wave function is defined from the molecule object. We only consider here the
ground state of the molecule in the CI expansion.

>>> # define the wave function
>>> wf = SlaterJastrow(mol, kinetic='jacobi',
>>>          configs='ground_state',
>>>           use_jastrow=True)

We use a Metropolis Hasting sampler with 100 walkers each performing 2000 steps.

>>> # sampler
>>> sampler = Metropolis(nwalkers=1000, nstep=2000, step_size=0.5,
>>>                      nelec=wf.nelec, ndim=wf.ndim,
>>>                      init=mol.domain('normal'),
>>>                      move={'type': 'one-elec', 'proba': 'normal'})

As an opimizer we use ADAM and define a simple linear scheduler.

>>> # optimizer
>>> opt = Adam(wf.parameters(), lr=0.005)
>>>
>>> # scheduler
>>> scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.75)

We can now assemble the solver

>>> # solver
>>> solver = SolverSlaterJastrow(wf=wf,
>>>                        sampler=sampler,
>>>                        optimizer=opt,
>>>                        scheduler=scheduler)

To optimize the geometry of the molecule we must specify `task=geo_opt`. This will
freeze all the parameters of the wave function at the exception of th atomic coordinate.
We can then run the optimization here using 50 epochs, the energy as loss function and the low variance expression
of the gradients.

>>> # optimize the geometry
>>> solver.configure(task='geo_opt')
>>> obs = solver.run(50, loss='energy', grad='manual')
>>> solver.save_traj('h2o_traj.xyz')

We can then plot the energy

>>> # plot the data
>>> plot_energy(obs.local_energy)

.. image:: ../../pics/h2_go_opt.png


