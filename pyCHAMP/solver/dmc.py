import numpy as np
from functools import partial
from pyCHAMP.solver.solver_base import SOLVER_BASE

class DMC(SOLVER_BASE):

	def __init__(self,wf=None, sampler=None, optimizer=None):
		SOLVER_BASE.__init__(self,wf,sampler,optimizer)

	def sample(self,param):
		
		wf_func = partial(self.wf.values,param)
		self.sampler.set_wf(wf_func)

		drift_func = partial(self.wf.drift_fd,param)
		self.sampler.set_drift_func(drift_func)

		energy_func = partial(self.wf.local_energy,param)
		self.sampler.set_energy_func(energy_func)

		pos = self.sampler.generate()
		return pos