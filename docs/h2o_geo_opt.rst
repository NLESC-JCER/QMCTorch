Geometry Optimization : H2O
====================================

We present here a complete example on how to use QMCTorch on a H2O molecule

>>> from torch import optim
>>> from torch.optim import Adam
>>>  from qmctorch.wavefunction import Orbital
>>> from qmctorch.solver import SolverOrbital
>>> from qmctorch.samplerimport Metropolis
>>> from qmctorch.wavefunction import Molecule

>>> from qmctorch.utils import plot_energy

>>> # define the molecule
>>> mol = Molecule(atom='water.xyz', unit='angs',
>>>                calculator='pyscf', basis='sto-3g')
>>> 
>>> # define the wave function
>>> wf = Orbital(mol, kinetic='jacobi',
>>>          configs='single(2,2)',
>>>           use_jastrow=True)

>>> # sampler
>>> sampler = Metropolis(nwalkers=1000, nstep=2000, step_size=0.5,
>>>                      nelec=wf.nelec, ndim=wf.ndim,
>>>                      init=mol.domain('normal'),
>>>                      move={'type': 'one-elec', 'proba': 'normal'})
>>> 
>>> 
>>> # optimizer
>>> opt = Adam(wf.parameters(), lr=0.005)
>>> 
>>> # scheduler
>>> scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.75)
>>> 
>>> # solver
>>> solver = SolverOrbital(wf=wf,
>>>                        sampler=sampler,
>>>                        optimizer=opt,
>>>                        scheduler=scheduler)
>>> 
>>> 
>>> 
>>> # optimize the geometry
>>> solver.configure(task='geo_opt')
>>> solver.track_observable(['local_energy','atomic_distances'])
>>> solver.run(5,loss='energy')
>>> solver.save_traj('h2o_traj.xyz')

>>> # plot the data
>>> plot_energy(solver.obs_dict)


.. image:: ../pics/h2_go_opt.png