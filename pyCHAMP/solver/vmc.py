import numpy as np 
from functools import partial

class VMC(object):

	def __init__(self,wf=None, sampler=None, optimizer=None):

		self.wf = wf
		self.sampler = sampler
		self.optimizer = optimizer	

		self.history = {'eneregy':[],'variance':[],'param':[]}

		if sampler is not None:
			self.sampler.set_ndim(wf.ndim)			

		if optimizer is not None:
			self.optimizer.func = self.wf.energy
			self.optimizer.grad = self.wf.energy_gradient

	def sample(self,param):
		partial_pdf = partial(self.wf.pdf,param)
		pos = self.sampler.generate(partial_pdf)
		return pos

	def energy(self,param,pos):
		return self.wf.energy(param,pos)

	def variance(self,param,pos):
		return self.wf.variance(param,pos)

	def optimize(self,init_param):

		param = init_param
		self.history['energy'] = []
		self.history['variance'] = []
		self.history['param'] = []
		for i in range(self.optimizer.maxiter):

			pos = self.sample(param)
			e = self.energy(param,pos)
			s = self.variance(param,pos)
			print('%d energy = %f, variance = %f (beta=%f)' %(i,e,s,param[0]))

			param, success = self.optimizer.update_parameters(param,pos)

			self.history['energy'].append(e)
			self.history['variance'].append(s)
			self.history['param'].append(param)

			if success:
				print('Optimization Done')
				break

		return success

