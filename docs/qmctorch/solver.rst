Solvers
=========================

Solvers are responsibe to orchestrate the calculations by combining the different elements, Molecule, QMCNet wave function, samplers and optimizers/schedulers
The main solver is caled `SolverSlaterJastrow` and is defined as

>>> from qmctorch.solver import SolverSlaterJastrow
>>> solver = SolverSlaterJastrow(wf=wf, sampler=sampler,
>>>                       optimizer=opt, scheduler=scheduler, output='output.hdf5'

As soon as the solver its content is defined the HDF5 file specficied byt `out`. This file will contain all the parameter
of the solver and can be explored using the dedicated `h5x` browser. This solver allows to perform different calculations as detailled below

Single point calculation
----------------------------

A single point calculation sample the current wave function and computes the energy & variance of the system.
It can simply be done by:

>>> obs = solver.single_point()
>>> plt.hist(obs.local_energy)

`obs` is a `SimpleNamespace` instance with the following attributes:
  * `obs.pos` : position of the walkers
  * `obs.local_energy` : values of the local energies for each sampling point
  * `obs.energy` : energy of the systems (i.e. the mean of the local energy)
  * `obs.variance` : variance of the local energy values
  * `obs.error` : error on the energy

The result of the calculation will also be stored in the hdf5 output file. The energy distribution can be vizualised
with the matplotlib histogram function.

Sampling trajectory
----------------------------

It is possible to compute the local energy during the propagation of the wlakers to assess the convergence of the sampling
To this end the sampler must be configured to output the walker position after each `N` steps.
For example to start recording the walkers positions after 1000 MC steps and then record their position each 100 MC steps one can use :

>>> from qmctorch.utils import plot_walkers_traj
>>> solver.sampler.ntherm = 1000
>>> solver.sampler.ndecor = 100
>>> pos = solver.sampler(solver.wf.pdf)
>>> obs = solver.sampling_traj(pos)
>>> plot_walkers_traj(obs.local_energy)

There as well the results are returned in the `obs` SimpleNamespace and are stored in the hdf5 file.
The trajectory can be visualized with the `plot_wakers_traj` routine of QMCTorch

Wave function optimization
-------------------------------

Optimizing the wave function is the main task of the solver. Before otpimization starts the solver needs to be
configured properly.

>>> solver.configure(task='wf_opt', freeze=['ao', 'mo'])

To main task are available wave function optimization (`wf_opt`) and geometry optimization (`geo_opt`).
If a wave function optimization is selected the atom coordinate will be frozen while all the other parameters of the QMCNet will be optimized.
If a geometry optimization is selected only the atom coordinates will be optimized. One cal also freeze (i.e. not optimize) certain parameter groups.
In the example above the parameters of the atomic orbitals and molecular orbitals will be frozen,

One can specified the observale that needs to be recorded during the optimization.

>>> solver.track_observable(['local_energy'])

By default the local energy and all the variational parameters will be recorded.

As the system is optimized, one can resample the wave function by changing the positions of the walkers.
Several strategies are available to resample the wave function. The preferred one is to update the walkers by performing a small number of MC steps after each optimization step.
This can be specified with :

>>> solver.configure_resampling(mode='update', resample_every=1, nstep_update=25)

Finally we can now optimize the wave function using the `.run()` method of the solver.
This methods takes a few arguments, the number of optimization step, the batchsize, and some parameters to compute the gradients.

>>> data = solver.run(5, batchsize=None,
>>>                  loss='energy',
>>>                  grad='manual',
>>>                  clip_loss=False)

The results are returned in a SimpleNamespace and can be visualized with dedicated routines :

>>> plot_energy(solver.observable.local_energy, e0=-
>>>              1.1645, show_variance=True)

>>> plot_data(solver.observable, obsname='jastrow.weight')