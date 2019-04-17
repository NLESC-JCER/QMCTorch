from pyCHAMP.optimizer.opt_base import OPT_BASE
from scipy.optimize import minimize

class MINIMIZE(OPT_BASE):

	def __init__(self, method=None, maxiter=100, tol=1E-6, x0=None):

		self.method = method
		self.tol = tol
		self.parameters = x0
		self.maxiter = maxiter

		self.func = None
		self.grad = None

	def update_parameters(self,param,pos):
		opt = {'maxiter' : 1, 'disp' : False}
		res = minimize(self.func, 
					   param, 
					   args = pos,
					   method = self.method, 
					   options = opt,
					   tol = self.tol, 
					   jac = self.grad)
		return res.x, res.success


