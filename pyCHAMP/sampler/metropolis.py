import numpy as np 
from pyCHAMP.sampler.sampler_base import SAMPLER_BASE

class METROPOLIS(SAMPLER_BASE):

	def __init__(self, nwalkers=1000, nstep=1000, mc_step_size = 3, boundary = 2):
		''' METROPOLIS HASTING SAMPLER
		Args:
			f (func) : function to sample
			nstep (int) : number of mc step
			nwalkers (int) : number of walkers
			eps (float) : size of the mc step
			boudnary (float) : boudnary of the space
		'''

		self.nwalkers = nwalkers
		self.nstep = nstep
		self.mc_step_size = mc_step_size
		self.boundary = boundary
		self.pdf = None
		self.initial_guess = None

	def generate(self):

		''' perform a MC sampling of the function f
		Returns:
			X (list) : position of the walkers
		'''

		if self.initial_guess is None:
			X = self.boundary * (-1 + 2*np.random.rand(self.nwalkers))
		else:
			X = self.initial_guess

		fx = self.pdf(X)

		ones = np.ones(self.nwalkers)	

		for istep in range(self.nstep):

			# new position
			xn =  X + self.mc_step_size * (2*np.random.rand(self.nwalkers) - 1)	

			# new function
			fxn = self.pdf(xn)
			df = fxn/fx

			# probability
			P = np.minimum(ones,df)
			tau = np.random.rand(self.nwalkers)

			# update
			index = P-tau>=0
			X[index] = xn[index]
			fx[index] = fxn[index]
		
		return X