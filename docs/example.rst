Wave function Optimization : H2
====================================

We present here a complete example on how to use QMCTorch on a H2 molecule

>>> from torch import optim
>>> from qmctorch.wavefunction import Orbital, Molecule
>>> from qmctorch.solver import SolverOrbital
>>> from qmctorch.sampler import Metropolis
>>> 
>>> from qmctorch.utils import set_torch_double_precision
>>> from qmctorch.utils import (plot_energy, plot_data, plot_walkers_traj)
>>> 
>>> set_torch_double_precision()
>>> 
>>> # define the molecule
>>> mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
>>>                calculator='adf',
>>>                basis='dzp',
>>>                unit='bohr')
>>> 
>>> # define the wave function
>>> wf = Orbital(mol, kinetic='jacobi',
>>>              configs='cas(2,2)',
>>>              use_jastrow=True)
>>> 
>>> # define the sample 
>>> sampler = Metropolis(nwalkers=500,
>>>                      nstep=2000, step_size=0.2,
>>>                      ntherm=-1, ndecor=100,
>>>                      nelec=wf.nelec, init=mol.domain('atomic'),
>>>                      move={'type': 'all-elec', 'proba': 'normal'})
>>> 
>>> # optimizer
>>> lr_dict = [{'params': wf.jastrow.parameters(), 'lr': 1E-3},
>>>            {'params': wf.ao.parameters(), 'lr': 1E-6},
>>>            {'params': wf.mo.parameters(), 'lr': 1E-3},
>>>            {'params': wf.fc.parameters(), 'lr': 1E-3}]
>>> opt = optim.Adam(lr_dict, lr=1E-3)
>>> 
>>> # scheduler
>>> scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.90)
>>> 
>>> # QMC solver
>>> solver = SolverOrbital(wf=wf, sampler=sampler, optimizer=opt, scheduler=None)
>>> 
>>> # perform a single point calculation
>>> # obs = solver.single_point()
>>> 
>>> # compute the sampling trajectory
>>> solver.sampler.ntherm = 1000
>>> solver.sampler.ndecor = 100
>>> pos = solver.sampler(solver.wf.pdf)
>>> obs = solver.sampling_traj(pos)
>>> plot_walkers_traj(obs.local_energy)
>>> solver.sampler.ntherm = -1
>>> 
>>> # optimize the wave function
>>> solver.configure(task='wf_opt', freeze=['ao', 'mo'])
>>> solver.track_observable(['local_energy'])
>>> 
>>> solver.configure_resampling(mode='update', resample_every=1, nstep_update=25)
>>> data = solver.run(5, batchsize=None, loss='energy', grad='manual', clip_loss=False)
>>> 
>>> plot_energy(solver.observable.local_energy, e0=-1.1645, show_variance=True)
>>> plot_data(solver.observable, obsname='jastrow.weight')
>>> 
>>> # optimize the geometry
>>> solver.configure(task='geo_opt')
>>> solver.tack_observable(['local_energy', 'atomic_distances'])
>>> data = solver.run(5, batchsize=None, loss='energy', grad='manual', clip_loss=False)

.. image:: ../pics/h2_dzp.png