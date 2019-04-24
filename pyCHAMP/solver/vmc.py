import numpy as np 
from functools import partial
from pyCHAMP.solver.solver_base import SOLVER_BASE

class VMC(SOLVER_BASE):

	def __init__(self, wf=None, sampler=None, optimizer=None):
		SOLVER_BASE.__init__(self,wf,sampler,optimizer)
		
	def sample(self,param):
		partial_pdf = partial(self.wf.pdf,param)
		pos = self.sampler.generate(partial_pdf)
		return pos


