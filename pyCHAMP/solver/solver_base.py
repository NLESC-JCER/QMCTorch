import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SOLVER_BASE(object):

	def __init__(self,wf=None, sampler=None, optimizer=None):

		self.wf = wf
		self.sampler = sampler
		self.optimizer = optimizer	

		self.history = {'eneregy':[],'variance':[],'param':[]}

		if optimizer is not None:
			self.optimizer.func = self.wf.energy
			self.optimizer.grad = self.wf.energy_gradient

	def sample(self,param):
		raise NotImplementedError('Implement the sample method of the solver')

	def energy(self,param,pos):
		return self.wf.energy(param,pos)

	def variance(self,param,pos):
		return self.wf.variance(param,pos)

	def single_point(self,param):
		pos = self.sample(param)
		e = self.energy(param,pos)
		s = self.variance(param,pos)
		return pos, e, s

	def plot_density(self,pos):

		if self.wf.ndim == 1:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			if self.wf.nelec == 1:
				plt.hist(pos)
			else:
				for ielec in range(self.wf.nelec):
					plt.hist(pos[ielec,:])

		elif self.wf.ndim == 2:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			for ielec in range(self.wf.nelec):
				plt.scatter(pos[:,ielec*2],pos[:,ielec*2+1])

		elif self.wf.ndim == 3:
			fig = plt.figure()
			ax = fig.add_subplot(111,projection='3d')
			for ielec in range(self.wf.nelec):
				ax.scatter(pos[:,ielec*3],pos[:,ielec*3+1],pos[:,ielec*3+2])
		plt.show()

	def plot_history(self):

		plt.plot(self.history['energy'])
		plt.plot(self.history['variance'])
		plt.show()

	def optimize(self,init_param):

		param = init_param
		self.history['energy'] = []
		self.history['variance'] = []
		self.history['param'] = []

		for i in range(self.optimizer.maxiter):

			pos = self.sample(param)
			e = self.energy(param,pos)
			s = self.variance(param,pos)

			param, success = self.optimizer.update_parameters(param,pos)
			print('%d energy = %f, variance = %f (beta=%f)' %(i,e,s,param[0]))

			self.history['energy'].append(e)
			self.history['variance'].append(s)
			self.history['param'].append(param)

			if success:
				print('Optimization Done')
				break

		return success