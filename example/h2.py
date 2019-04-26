import autograd.numpy as np

from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.optimizer.minimize import MINIMIZE
from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.sampler.hamiltonian import HAMILTONIAN
from pyCHAMP.solver.vmc import VMC

class ORBITAL_1S(object):

	def __init__(self,pos,beta):
		self.pos = pos
		self.beta = beta

	def val(self,epos):
		r = np.sqrt(np.sum((epos-self.pos)**2,1))
		return np.exp(-self.beta*r)

class H(object):

	def __init__(self,pos):

		self.name = 'H'
		self.basis = 'minimal'
		self.pos = pos
		self.Z = 1
		self.orbs = [ORBITAL_1S(pos,1.)]


class H2(WF):

	def __init__(self,nelec,ndim):
		
		WF.__init__(self, nelec, ndim)

		self.atoms = [ H([0,0,-1]),H([0,0,1])]
		self.natom = len(self.atoms)
		

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

		SD = np.zeros((pos.shape[0],self.nelec,self.nelec))
		for ielec in range(self.nelec):

			epos = pos[:,ielec*self.ndim:(ielec+1)*self.ndim]

			iorb = 0
			for at in self.atoms:
				for orb in at.orbs:
					SD[:,ielec,iorb] = orb.val(epos)
					iorb += 1

		return np.linalg.det(SD).reshape(-1,1)

	def nuclear_potential(self,pos):
		
		nwalker = pos.shape[0]
		pot = np.zeros((nwalker,1))

		for ielec in range(self.nelec):
			epos = pos[:,ielec*self.ndim:(ielec+1)*self.ndim]
			for at in self.atoms:
				r = np.sqrt(np.sum((epos-at.pos)**2,1))
				pot -= (at.Z / r).reshape(-1,1)

		return pot.reshape(-1,1)

	def electronic_potential(self,pos):

		nwalker = pos.shape[0]
		pot = np.zeros((nwalker,1))

		for ielec1 in range(self.nelec-1):
			epos1 = pos[:,ielec1*self.ndim:(ielec1+1)*self.ndim]

			for ielec2 in range(ielec1+1,self.nelec):
				epos2 = pos[:,ielec2*self.ndim:(ielec2+1)*self.ndim]
				r = np.sqrt(np.sum((epos1-epos2)**2,1))

				pot -= (1./r).reshape(-1,1)

		return pot


if __name__ == "__main__":

	wf = H2(nelec=2, ndim=3)
	sampler = METROPOLIS(nwalkers=1000, nstep=1000, step_size = 3, nelec=2, ndim=3, domain = {'min':-5,'max':5})
	#sampler = HAMILTONIAN(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=3)
	#optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)

	# VMS solver
	vmc = VMC(wf=wf, sampler=sampler, optimizer=None)

	# single point
	opt_param = [1.]
	pos,e,s = vmc.single_point(opt_param)
	print('Energy   : ', e)
	print('Variance : ', s)
	vmc.plot_density(pos)



	# # optimization
	# init_param = [0.5]
	# vmc.optimize(init_param)
	# vmc.plot_history()
	

