import numpy as np 
import torch
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from functools import partial
from pyCHAMP.solver.solver_base import SOLVER_BASE



class QMC_DataSet(Dataset):

	def __init__(self, data):
		self.data = data
	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self,index):
		return self.data[index,:]

class QMCLoss(nn.Module):

	def __init__(self,wf,method='energy'):

		super(QMCLoss,self).__init__()
		self.wf = wf
		self.method = method

	def forward(self,param,input_pos):

		ndim = input_pos.shape[1]
		pos = input_pos.view(-1,ndim).data.numpy()
		pos = pos.T

		if self.method == 'variance':
			l = self.wf.variance(param,pos)
		elif self.method == 'energy':
			l = self.wf.energy(param,pos)
		return Variable(l)
			

class NN(SOLVER_BASE):

	def __init__(self, wf=None, sampler=None, model=None, optimizer=None):

		SOLVER_BASE.__init__(self,wf,sampler,None)
		self.net = model()
		self.opt = optim.SGD(self.net.parameters(),lr=0.005, momentum=0.9, weight_decay=0.001)
		self.batchsize = 32


	def sample(self,param):

		partial_pdf = partial(self.wf.pdf,param)
		pos = self.sampler.generate(partial_pdf)
		return pos

	def super_sample(self,param,n=100):

		pos = []
		for _ in range(n):
			pos.append(self.sample(param).T)
		return np.array(pos)

	def train(self,nepoch,param):

		pos = self.super_sample(param)
		param = Variable(torch.tensor(param))

		dataset = QMC_DataSet(pos)
		dataloader = DataLoader(dataset,batch_size=self.batchsize)
		qmc_loss = QMCLoss(self.wf,method='energy')
		
    
		for n in range(nepoch):

			for data in dataloader:
				
				data = Variable(data).float()
				dp = self.net(data)
				print(dp.shape)
				param += dp

				loss = qmc_loss(param,data)
				self.opt.zero_grad()
				loss.backward()
				self.opt.step()

				pos = self.super_sample(param)
				dataloader.dataset.data = pos


if __name__ == "__main__":

	from pyCHAMP.solver.vmc import VMC
	from pyCHAMP.wavefunction.wf_base import WF
	from pyCHAMP.sampler.metropolis import METROPOLIS
	from pyCHAMP.optimizer.PointNet import PointNetCls, PointNetfeat

	class HarmOsc3D(WF):

		def __init__(self,nelec,ndim):
			WF.__init__(self, nelec, ndim)

		def values(self,parameters,pos):
			''' Compute the value of the wave function.

			Args:
				parameters : variational param of the wf
				pos: position of the electron

			Returns: values of psi
			# '''
			# if pos.shape[1] != self.ndim :
			# 	raise ValueError('Position have wrong dimension')

			beta = parameters[0]
			pos = pos.T
			return np.exp(-beta*pos[0]**2)*np.exp(-beta*pos[1]**2)*np.exp(-beta*pos[2]**2) 

		def nuclear_potential(self,pos):
			return np.sum(0.5*pos**2,1)

		def electronic_potential(self,pos):
			return 0

	wf = HarmOsc3D(nelec=1, ndim=2)
	sampler = METROPOLIS(nwalkers=100, nstep=100, 
		                 step_size = 3, nelec=1, 
		                 ndim=3, domain = {'min':-2,'max':2})

	nn = NN(wf=wf,sampler=sampler,model=PointNetfeat)
	nn.train(100,[1.0])