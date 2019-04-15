import numpy as np 

class VMC(object):

	def __init__(self,wf=None, sampler=None, optimizer=None):

		self.wf = wf
		self.sampler = sampler
		self.opt = optimizer		

	def evaluate(self):
		self.sampler.set_pdf(self.wf.get)
		pos = self.sampler.generate()
		return self._total_energy(pos)


	def _local_energy(self,pos):
		return self.wf.applyH(pos) / self.wf.get(pos)

	def _total_energy(self,pos):
		nsample = pos.shape[0]
		return 1./nsample * np.sum(self._local_energy(pos))