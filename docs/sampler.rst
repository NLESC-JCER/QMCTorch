Sampler in QMCTorch
--------------------------------

`QMCTorch` offers different sampler to propagate the walkers. The defualt sampler is a Metropolis-Hasting 
that can be defined as follows :

>>> from qmctorch.sampler import Metropolis
>>> sampler = Metropolis(nwalkers=500,
>>>                  nstep=2000, step_size=0.2,
>>>                  ntherm=-1, ndecor=100,
>>>                  nelec=wf.nelec, init=mol.domain('atomic'),
>>>                  move={'type': 'all-elec', 'proba': 'normal'})

