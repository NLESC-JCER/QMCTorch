from pyCHAMP.optimizer.opt_base import OPT_BASE
from pyswarm import pso
import numpy as np 

class SWARM(OPT_BASE):

	def __init__(self, method=None, maxiter=100, tol=1E-6):

		self.method = method
		self.tol = tol
		self.maxiter = maxiter

		self.func = None
		self.grad = None

	def update_parameters(self,param,pos):
		
		lower_bound = [param[0] - 0.1]
		upper_boud = [param[0] + 0.1]
		xopt, fopt = pso(self.func, 
						 lower_bound, 
						 upper_boud,
						 args = (pos,),
						 swarmsize = 100,
						 maxiter=100)	
		return xopt, False