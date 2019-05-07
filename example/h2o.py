from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.wavefunction.wf_pyscf import PYSCF_WF
from pyCHAMP.solver.vmc import VMC


wf = PYSCF_WF(atom='O 0 0 0; H 0 0 1; H 0 1 0',basis='dzp')
sampler = METROPOLIS(nwalkers=100, nstep=100, step_size = 3, 
                     nelec=wf.nelec, ndim=wf.ndim, 
                     domain = {'min':-5,'max':5})

param = []
vmc = VMC(wf=wf, sampler=sampler, optimizer=None)
pos = vmc.sample(param)
vmc.plot_density(pos)
e = vmc.wf.energy(param,pos)