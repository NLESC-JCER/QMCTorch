import autograd.numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
from functools import partial
from pyCHAMP.optimizer.minimize import MINIMIZE
from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.vmc.vmc import VMC

class HarmOsc1D(WF):

	def __init__(self,nelec,ncart,derivative,parameters=None):
		WF.__init__(self, nelec, ncart, parameters,derivative)

	def value(self,parameters,pos):
		''' Compute the value of the wave function.

		Args:
			parameters : parameters of th wf
			x: position of the electron

		Returns: values of psi
		'''
		beta = parameters[0]
		return np.exp(-beta*pos**2)

	def get(self,pos):
		return partial(self.value,self.parameters)(pos)

	def fp(self,parameters):
		return partial(self.value,pos=self.pos)(parameters)

	def grad_opt(self,parameters):
		return egrad(self.fp)(parameters)
		
	def manual_grad(self,parameters,pos):
		beta = parameters[0]
		return -2*beta*pos*np.exp(-beta*pos**2)

	def nuclear_potential(self,pos):
		return 0.5*pos**2 

	def electronic_potential(self,pos):
		return 0


wf = HarmOsc1D(nelec=1,ncart=1,parameters=[0.5], derivative='autograd')
sampler = METROPOLIS(nwalkers=1000, nstep=1000, mc_step_size = 3, boundary = 2)
optimizer = MINIMIZE(method='Nelder-Mead')

vmc = VMC(wf=wf, sampler=sampler, optimizer=optimizer)

pos = vmc.sample()
e = vmc.energy(pos)
v = vmc.variance(pos)
print('Energy : ', e)
print('Variance : ', v)
plt.hist(pos)
plt.show()

#res = vmc.optimize([1.0])


