import autograd.numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
from pyCHAMP.optimizer.minimize import MINIMIZE
from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.vmc.vmc import VMC

class HarmOsc1D(WF):

	def __init__(self,nelec,ncart):
		WF.__init__(self, nelec, ncart)

	def values(self,parameters,pos):
		''' Compute the value of the wave function.

		Args:
			parameters : parameters of th wf
			x: position of the electron

		Returns: values of psi
		'''
	
		beta = parameters[0]
		return np.exp(-beta*pos**2)

	def nuclear_potential(self,pos):
		return 0.5*pos**2 

	def electronic_potential(self,pos):
		return 0

opt_param = [0.5]
wf = HarmOsc1D(nelec=1,ncart=1)
sampler = METROPOLIS(nwalkers=1000, nstep=1000, mc_step_size = 3, boundary = 2)
optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)

vmc = VMC(wf=wf, sampler=sampler, optimizer=optimizer)

x0 = [1.25]
vmc.optimize(x0)

plt.plot(vmc.history['energy'])
plt.plot(vmc.history['variance'])
plt.show()


