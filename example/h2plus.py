import autograd.numpy as np

from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.optimizer.minimize import MINIMIZE
from pyCHAMP.sampler.metropolis import METROPOLIS

from pyCHAMP.solver.vmc import VMC

from pyCHAMP.solver.mesh import regular_mesh_3d

import matplotlib.pyplot as plt

class Hydrogen(WF):

	def __init__(self,nelec,ndim):
		
		WF.__init__(self, nelec, ndim)

		x = 0.97
		self.x1 = np.array([0,0,-x])
		self.x2 = np.array([0,0, x])

	def _gaussian_values(self,parameters,pos):
		''' Compute the value of the wave function.

		Args:
			parameters : parameters of th wf
			x: position of the electron

		Returns: values of psi
		'''
		beta = parameters[0]
		if pos.ndim == 1:
			pos = pos.reshape(1,-1)

		k = np.sqrt( (2*np.pi*beta)**3 )

		r1 = np.sum((pos-self.x1)**2,1)
		r2 = np.sum((pos-self.x2)**2,1)

		p1 = 2./k * np.exp(-0.5*beta*r1).reshape(-1)
		p2 = 2./k * np.exp(-0.5*beta*r2).reshape(-1)

		return p1+p2

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


		r1 = np.sqrt(np.sum((pos-self.x1)**2,1))
		r2 = np.sqrt(np.sum((pos-self.x2)**2,1))
		
		p1 = 2. * np.exp(-beta*r1).reshape(-1)
		p2 = 2. * np.exp(-beta*r2).reshape(-1)

		return p1+p2

	def nuclear_potential(self,pos):
		
		r1 = np.sqrt(np.sum((pos-self.x1)**2,1))
		r2 = np.sqrt(np.sum((pos-self.x2)**2,1))

		rm1 = - 1./ r1 - 1. / r2
		return rm1.reshape(-1)

	def electronic_potential(self,pos):
		return 0

	def nuclear_repulsion(self):
		rnn = np.sqrt(np.sum((self.x1-self.x2)**2))
		return 1./rnn


if __name__ == "__main__":

	wf = Hydrogen(nelec=1, ndim=3)
	sampler = METROPOLIS(nwalkers=5000, nstep=1000, step_size = 3, nelec=1, ndim=3, domain = {'min':-5,'max':5})
	#sampler = HAMILTONIAN(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=3)
	optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)

	# VMC solver
	vmc = VMC(wf=wf, sampler=sampler, optimizer=optimizer)

	energy,var = [],[]
	opt_param = [1.0]

	X = np.linspace(0.1,2,25)
	for x in X:
		vmc.wf.x1 = np.array([0,0,-x])
		vmc.wf.x2 = np.array([0,0,x])
		vnn = wf.nuclear_repulsion()
		pos,e,s = vmc.single_point(opt_param)	
		energy.append(e+vnn)
		var.append(np.var(vmc.wf.local_energy(opt_param,pos)+vnn))

	plt.plot(X,energy)
	plt.show()

	plt.plot(X,var)
	plt.show()

	# single point
	opt_param = [1.0]
	pos,e,s = vmc.single_point(opt_param)
	print('Energy   : ', e)
	print('Variance : ', s)
	vmc.plot_density(pos)

	pts = regular_mesh_3d(xmin=-2,xmax=2,ymin=-2.,ymax=2,zmin=-5,zmax=5,nx=5,ny=5,nz=5)
	pos = np.array(pts)

	exit()
	# optimization
	init_param = [0.5]
	vmc.optimize(init_param)
	vmc.plot_history()
	

