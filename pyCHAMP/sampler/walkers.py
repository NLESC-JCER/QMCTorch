import numpy as np 

class WALKERS(object):

	def __init__(self,nwalkers,nelec,ndim,domain):

		self.nwalkers = nwalkers
		self.ndim = ndim
		self.nelec = nelec
		self.domain = domain

		self.pos = None
		self.status = None

	def initialize(self, method='uniform', pos=None):

		if pos is not None:
			self.pos = pos
		
		else:
			options = ['center','uniform']
			if method not in options:
				raise ValueError('method %s not recognized. Options are : %s ' %(method, ' '.join(options)) )

			if method == options[0]:
				self.pos = np.zeros((self.nwalkers, self.nelec*self.ndim )) + eps

			elif method == options[1]:
				self.pos = np.random.rand(self.nwalkers, self.nelec*self.ndim) 
				self.pos *= self.domain['max'] - self.domain['min']
				self.pos += self.domain['min']

		self.status = np.ones((self.nwalkers,1))

	def move(self, step_size, method='one'):

		if method == 'one':
			new_pos = self._move_one(step_size,index)

		elif method == 'all':
			new_pos = self._move_all(step_size)

		return new_pos

	def _move_all(self,step_size):
		return self.pos + self.status * self._random(step_size,(self.nwalkers,self.nelec * self.ndim))

	def _move_one(self,step_size):

		new_pos = np.copy(self.pos)
		ielec =  np.random.ranint(0,self.nelec,self.nwalkers)
		for iw in range(self.nwalkers):
			if self.status[iw] == 1:
				index = [self.ndim*ielec[iw],self.ndim*(ielec[iw]+1)]
				new_pos[iw,index[0]:index[1]] += self._random(step_size,(self.ndim,))	
		return new_pos

	def _random(self,step_size,size):
		return step_size * (2 * np.random.random(size) - 1)











