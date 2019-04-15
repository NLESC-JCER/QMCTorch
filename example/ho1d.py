import autograd.numpy as np
import matplotlib.pyplot as plt
from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.vmc.vmc import VMC

class HarmOsc1D(WF):

	def __init__(self,nelec,ncart,derivative,parameters=None):
		WF.__init__(self, nelec, ncart, parameters,derivative)

	def get(self,pos):
		''' Compute the value of the wave function.

		Args:
			x: position of the electron
			kwargs: argument of the wf

		Returns: values of psi
		'''
		beta = self.parameters['beta']
		return np.exp(-beta*pos**2)

	def nuclear_potential(self,pos):
		return 0.5*pos**2 

	def electronic_potential(self,pos):
		return 0


wf = HarmOsc1D(nelec=1,ncart=1,parameters={'beta' : 0.5}, derivative='autograd')
sampler = METROPOLIS(nwalkers=1000, nstep=1000, mc_step_size = 3, boundary = 2)


vmc = VMC(wf=wf, sampler=sampler, optimizer=None)
e,psi = vmc.evaluate()
print(e)

plt.hist(psi)
plt.show()


