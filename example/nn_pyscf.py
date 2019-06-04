from pyCHAMP.solver.vmc import VMC
from pyCHAMP.wavefunction.neural_pyscf_wf_base import NEURAL_PYSCF_WF
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS
from pyCHAMP.solver.neural_net import NN4PYSCF
import matplotlib.pyplot as plt
wf = NEURAL_PYSCF_WF(atom='H 0 1 0; H 0 0 1',
                     #basis='dzp',
                     basis='sto-3g',
                     active_space=(1,1))

sampler = METROPOLIS(nwalkers=100, nstep=100, 
                     step_size = 3., nelec = wf.nelec, 
                     ndim = 3, domain = {'min':-5,'max':5})

net = NN4PYSCF(wf=wf,sampler=sampler)

pos = net.sample(ntherm=0)
pos = pos.reshape(100,100,6)
var = net.observalbe(net.wf.variance,pos)
plt.plot(var)


exit()
net.wf.layer_mo.weight.data = torch.eye(net.wf.nao).double()
for param in net.wf.layer_ci.parameters():
    param.requires_grad = False


pos = net.sample(ntherm=0)
pos = pos.reshape(100,100,6)
var_ = net.observalbe(net.wf.variance,pos)
plt.plot(var_)
plt.show()

net.train(50,pos=pos,ntherm=-1)