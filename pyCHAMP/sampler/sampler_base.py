
class SAMPLER_BASE(object):

	def __init__(self):
		pass

	def set_ndim(self,ndim):
		self.ndim = ndim

	def set_initial_guess(self,guess):
		self.initial_guess = guess

	def generate(self):
		raise NotImplementedError()