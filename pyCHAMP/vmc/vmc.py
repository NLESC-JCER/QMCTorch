import numpy as np 

class VMC(object):

	def __init__(self,wf=None, sampler=None, optimizer=None):

		self.wf = wf
		self.sampler = sampler
		self.optimizer = optimizer		

		if sampler is not None:
			self.sampler.set_ndim(wf.ndim)
			self.sampler.set_pdf(self.wf.pdf)

		if optimizer is not None:
			self.optimizer.func = self.func_opt

	def sample(self):
		pos = self.sampler.generate()
		self.wf.pos = pos
		return pos

	def energy(self,pos):
		return self.wf.energy(pos)

	def variance(self,pos):
		return self.wf.variance(pos)

	def optimize(self,x0):
		self.optimizer.run(x0)

	def func_opt(self,parameters) :
		self.wf.parameters=parameters
		pos = self.sample()
		return self.energy(pos)