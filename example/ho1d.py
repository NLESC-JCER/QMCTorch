import autograd.numpy as np

from pyCHAMP.wavefunction.wf_base import WF

from pyCHAMP.optimizer.minimize import MINIMIZE
#from pyCHAMP.optimizer.swarm import SWARM

from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.sampler.pymc3 import PYMC3
from pyCHAMP.solver.vmc import VMC



class HarmOsc1D(WF):

	def __init__(self,nelec,ndim):
		WF.__init__(self, nelec, ndim)

	def values(self,parameters,pos):
		''' Compute the value of the wave function.

		Args:
			parameters : parameters of th wf
			x: position of the electron

		Returns: values of psi
		'''
	
		beta = parameters[0]
		return np.exp(-beta*pos**2).flatten()

	def nuclear_potential(self,pos):
		return (0.5*pos**2).flatten()

	def electronic_potential(self,pos):
		return 0

if __name__ == "__main__":

	wf = HarmOsc1D(nelec=1, ndim=1)
	sampler = METROPOLIS(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=1, domain = {'min':-2,'max':2})
	#sampler = PYMC3(nwalkers=1000,ndim=1)

	optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)

	
	# VMC solver
	vmc = VMC(wf=wf, sampler=sampler, optimizer=optimizer)

	# single point
	opt_param = [0.5]	
	pos,e,s = vmc.single_point(opt_param)
	print('Energy   : ', e)
	print('Variance : ', s)
	vmc.plot_density(pos)

	# optimization
	# init_param = [1.]
	# vmc.optimize(init_param)
	# vmc.plot_history()
	


