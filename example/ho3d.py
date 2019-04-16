import autograd.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.vmc.vmc import VMC

class HarmOsc3D(WF):

	def __init__(self,nelec,ncart,parameters=None):
		WF.__init__(self, nelec, ncart, parameters)

	def get(self,pos):
		''' Compute the value of the wave function.

		Args:
			x: position of the electron
			kwargs: argument of the wf

		Returns: values of psi
		'''
		if pos.shape[1] != self.ndim :
			raise ValueError('Position have wrong dimension')

		beta = self.parameters['beta']
		return np.prod(np.exp(-beta*pos**2),1).reshape(-1,1)

	def nuclear_potential(self,pos):
		return np.sum(0.5*pos**2,1).reshape(-1,1)

	def electronic_potential(self,pos):
		return 0


wf = HarmOsc3D(nelec=1, ncart=3,parameters={'beta' : 0.5})
sampler = METROPOLIS(nwalkers=1000, nstep=1000, mc_step_size = 1./3, boundary = 2)
#opt = MINIMIZE(method='bfgs',tol=1E-6)

vmc = VMC(wf=wf, sampler=sampler, optimizer=None)

pos = vmc.sample()
e = vmc.energy(pos)
v = vmc.variance(pos)
print('Evmc = ', e)
print('Vvmc = ', v)



fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(pos[:,0],pos[:,1],pos[:,2])
plt.show()
