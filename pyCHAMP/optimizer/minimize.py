from pyCHAMP.optimizer.opt_base import OPT_BASE
from scipy.optimize import minimize

class MINIMIZE():

	def __init__(self, method=None, tol=1E-6, maxiter=100):

		self.method = method
		self.tol = tol
		self.opt = {'maxiter' : maxiter, 'disp':True}
		self.func = None
		self.grad = None

	def run(self,x0):
		return minimize(self.func, x0, method=self.method, options=self.opt, tol=self.tol, jac=self.grad)
