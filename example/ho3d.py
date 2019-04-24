import autograd.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.optimizer.minimize import MINIMIZE
from pyCHAMP.solver.vmc import VMC

class HarmOsc3D(WF):

	def __init__(self,nelec,ndim):
		WF.__init__(self, nelec, ndim)

	def values(self,parameters,pos):
		''' Compute the value of the wave function.

		Args:
			parameters : variational param of the wf
			pos: position of the electron

		Returns: values of psi
		# '''
		# if pos.shape[1] != self.ndim :
		# 	raise ValueError('Position have wrong dimension')

		beta = parameters[0]
		return np.exp(-beta*np.sum(pos**2,1)).reshape(-1,1)

	def nuclear_potential(self,pos):
		return np.sum(0.5*pos**2,1).reshape(-1,1)

	def electronic_potential(self,pos):
		return 0

opt_param = [0.5]
wf = HarmOsc3D(nelec=1, ndim=3)
sampler = METROPOLIS(nwalkers=1000, nstep=1000, step_size=3, nelec=1, ndim=3, domain = {'min':-2,'max':2})
optimizer = MINIMIZE(method='bfgs', maxiter=20, tol=1E-4)

vmc = VMC(wf=wf, sampler=sampler, optimizer=optimizer)

x0 = [0.25]
vmc.optimize(x0)

plt.plot(vmc.history['energy'])
plt.plot(vmc.history['variance'])
plt.show()


