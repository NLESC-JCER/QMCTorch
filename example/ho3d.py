import autograd.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.optimizer.minimize import MINIMIZE
from pyCHAMP.vmc.vmc import VMC

class HarmOsc3D(WF):

	def __init__(self,nelec,ncart):
		WF.__init__(self, nelec, ncart)

	def value(self,parameters,pos):
		''' Compute the value of the wave function.

		Args:
			parameters : variational param of the wf
			pos: position of the electron

		Returns: values of psi
		'''
		if pos.shape[1] != self.ndim :
			raise ValueError('Position have wrong dimension')

		beta = parameters[0]
		return np.prod(np.exp(-beta*pos**2),1).reshape(-1,1)

	def nuclear_potential(self,pos):
		return np.sum(0.5*pos**2,1).reshape(-1,1)

	def electronic_potential(self,pos):
		return 0

opt_param = [0.5]
wf = HarmOsc3D(nelec=1, ncart=3)
sampler = METROPOLIS(nwalkers=1000, nstep=1000, mc_step_size = 1, boundary = 2)
optimizer = MINIMIZE(method='bfgs', maxiter=20, tol=1E-4)

vmc = VMC(wf=wf, sampler=sampler, optimizer=optimizer)

x0 = [0.25]
vmc.optimize(x0)

plt.plot(vmc.history['energy'])
plt.plot(vmc.history['variance'])
plt.show()

# pos = vmc.sample(opt_param)
# e = vmc.wf.energy(opt_param,pos)
# v = vmc.wf.variance(opt_param,pos)
# print('Evmc = ', e)
# print('Vvmc = ', v)

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(pos[:,0],pos[:,1],pos[:,2])
# plt.show()
