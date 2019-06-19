import autograd.numpy as np

from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.optimizer.minimize import MINIMIZE
from pyCHAMP.sampler.metropolis import METROPOLIS
#from pyCHAMP.sampler.hamiltonian import HAMILTONIAN
from pyCHAMP.solver.vmc import VMC

class Hydrogen(WF):

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
		if pos.ndim == 1:
			pos = pos.reshape(1,-1)
			
		r = np.sqrt(np.sum(pos**2,1))
		return 2*np.exp(-beta*r).reshape(-1)

	def nuclear_potential(self,pos):
		r = np.sqrt(np.sum(pos**2,1))
		rm1 = - 1./ r
		return rm1.reshape(-1)

	def electronic_potential(self,pos):
		return 0


if __name__ == "__main__":

	wf = Hydrogen(nelec=1, ndim=3)
	sampler = METROPOLIS(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=3, domain = {'min':-5,'max':5})
	#sampler = HAMILTONIAN(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=3)
	optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)

	# VMS solver
	vmc = VMC(wf=wf, sampler=sampler, optimizer=optimizer)

	# single point
	opt_param = [1.]
	pos,e,s = vmc.single_point(opt_param)
	print('Energy   : ', e)
	print('Variance : ', s)
	vmc.plot_density(pos)

	exit()
	# optimization
	init_param = [0.5]
	vmc.optimize(init_param)
	vmc.plot_history()
	

