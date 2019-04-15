import numpy as np 

class VMC(object):

	def __init__(self,wf=None, sampler=None, optimizer=None):

		self.wf = wf
		self.sampler = sampler
		self.opt = optimizer		

		self.sampler.set_ndim(wf.ndim)
		self.sampler.set_pdf(self.wf.pdf)

	def evaluate(self):
		pos = self.sampler.generate()
		return self._total_energy(pos), pos


	def _local_energy(self,pos):
		# print('Epot : ', np.sum(self.wf.nuclear_potential(pos)))
		# print('Ekin : ', np.sum(self.wf.kinetic(pos)))
		return self.wf.applyK(pos)/self.wf.get(pos) + self.wf.nuclear_potential(pos) + self.wf.electronic_potential(pos)

	def _total_energy(self,pos):
		nsample = pos.shape[0]
		return 1./nsample * np.sum(self._local_energy(pos))