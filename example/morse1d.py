import autograd.numpy as np

from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.optimizer.minimize import MINIMIZE
from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.solver.vmc import VMC

class Morse1D(WF):

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
		return np.exp(-np.exp(-beta*pos)-0.5*pos).reshape(-1,1)

	def nuclear_potential(self,pos):
		return 0.5*(np.exp(-2*pos) - 2*np.exp(-pos)) 

	def electronic_potential(self,pos):
		return 0


if __name__ == "__main__":

	wf = Morse1D(nelec=1, ndim=1)
	sampler = METROPOLIS(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=1, domain = {'min':-5,'max':15})
	optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)

	# VMS solver
	vmc = VMC(wf=wf, sampler=sampler, optimizer=optimizer)

	# single point
	opt_param = [1.]
	pos,e,s = vmc.single_point(opt_param)
	print('Energy   : ', e)
	print('Variance : ', s)
	vmc.plot_density(pos)

	# optimization
	init_param = [1.05]
	vmc.optimize(init_param)
	vmc.plot_history()
	



	# metro = METROPOLIS(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=1, domain = {'min':-2,'max':2})
	# optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)



	# vmc = VMC(wf=wf, sampler=metro, optimizer=optimizer)
	# pos = vmc.sample(opt_param)


	# diff = DIFFUSION(nwalkers=1000, nstep=1, step_size = 0.5, nelec=1, ndim=1, domain = {'min':-2,'max':2})
	# diff.set_initial_guess(pos)

	# dmc = DMC(wf=wf, sampler=diff, optimizer=None)
	# pos,e,s = dmc.single_point(opt_param)
	# dmc.plot_density(pos)




	# sampler = METROPOLIS(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=1, domain = {'min':-2,'max':2})
	# optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)
	# vmc = VMC(wf=wf, sampler=sampler, optimizer=optimizer)
	# x0 = [1.25]
	# vmc.optimize(x0)

	# plt.plot(vmc.history['energy'])
	# plt.plot(vmc.history['variance'])
	# plt.show()


