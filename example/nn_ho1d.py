import torch
from torch.autograd import Variable
from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF, WaveNet1D
from pyCHAMP.solver.neural_net import NN, QMCLoss
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS
import matplotlib.pyplot as plt

import numpy as np

wf = NEURAL_WF(ndim=1,nelec=1)

sampler = METROPOLIS(nwalkers=250, nstep=1000, 
                     step_size = 3., nelec = wf.nelec, 
                     ndim = wf.ndim, domain = {'min':-5,'max':5})

net = NN(wf=wf,sampler=sampler)
net.train(250,ntherm=-1,loss='variance')






# net.train(50,ntherm=-1)
# vals = net.wf(X)
# plt.plot(X.detach().numpy(),vals.detach().numpy())
# plt.show()


# pos = net.sample(ntherm=0)
# pos = pos.reshape(sampler.nwalkers,-1,wf.ndim_tot)
# var = net.observalbe(net.wf.variance,pos)
# plt.plot(var)
# plt.show()




# pos = net.sample(ntherm=0)
# pos = pos.reshape(100,100,6)
# var_ = net.observalbe(net.wf.variance,pos)
# plt.plot(var_)
# plt.show()

# net.train(50,pos=pos,ntherm=-1)