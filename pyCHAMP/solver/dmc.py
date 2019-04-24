import numpy as np

class DMC(object):

	def __init__(self,wf=None, diffuser=None, optimizer=None):

		self.wf = wf
		self.diffuser = diffuser
		self.optimizer = optimizer

		self.history = {'eneregy':[],'variance':[],'param':[]}


		if optimizer is not None:
			self.optimizer.func = self.wf.energy
			self.optimizer.grad = self.wf.energy_gradient