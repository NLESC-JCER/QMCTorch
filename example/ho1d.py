import numpy as np
from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.vmc.vmc import VMC

class HarmOsc1D(WF):

	def __init__(self,parameters=None):
		WF.__init__(self, parameters)

	def get(self,pos):
		''' Compute the value of the wave function.

		Args:
			x: position of the electron
			kwargs: argument of the wf

		Returns: values of psi
		'''
		beta = self.parameters['beta']
		norm = np.sqrt(np.pi/beta)
		return np.exp(-beta*pos**2)

	def nuclear_potential(self,pos):
		return 0.5*pos**2 * self.get(pos)

	def electronic_potential(self,pos):
		return 0


wf = HarmOsc1D(parameters={'beta' : 0.5})
sampler = METROPOLIS(nwalkers=1000, nstep=1000, mc_step_size = 3, boundary = 2)


vmc = VMC(wf=wf, sampler=sampler, optimizer=None)
e = vmc.evaluate()
print(e)


