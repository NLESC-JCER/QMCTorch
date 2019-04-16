import numpy as np
from scipy.optimize import minimize, rosen

class WF(object):

	def __init__(self):
		pass

	def f(self,opt,pos):
		return rosen(opt)


wf = WF()
opt = [1.3,0.7,0.8,1.9,1.2]
pos = np.random.rand(10,3)
res = minimize(wf.f,opt,args=(pos,),method='Nelder-Mead',tol=1E-6)
print(res.x)
