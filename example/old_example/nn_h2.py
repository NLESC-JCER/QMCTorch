import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from pyCHAMP.wavefunction.neural_pyscf_wf_base import NEURAL_PYSCF_WF
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS
from pyCHAMP.solver.deepqmc import DeepQMC
import matplotlib.pyplot as plt


# wavefunction 
# bond distance : 0.74 A -> 1.38 a
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree
wf = NEURAL_PYSCF_WF(atom='H 0 0 0; H 0 0 0.74',
                     basis='dz',
                     active_space=(1,1))
# sampler
sampler = METROPOLIS(nwalkers=1000, nstep=1000, 
                     step_size = 3., nelec = wf.nelec, 
                     ndim = 3, domain = {'min':-5,'max':5})

# optimizer
opt = optim.Adam(wf.parameters(),lr=0.005)
#opt = optim.SGD(wf.parameters(),lr=0.1)

# domain for the RBF Network
boundary = 5.
domain = {'xmin':-boundary,'xmax':boundary,
          'ymin':-boundary,'ymax':boundary,
          'zmin':-boundary,'zmax':boundary}
ncenter = [11,11,11]

# network
net = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)
obs_dict = {'local_energy':[],
            'atomic_distance':[],
            'get_sigma':[]}

if 1:
    pos = Variable(net.sample())
    pos.requires_grad = True
    e = net.wf.energy(pos)
    s = net.wf.variance(pos)

    print('Energy   :', e)
    print('Variance :', s)

if 0:

    X = np.linspace(0.5,2.5,25)
    energy, var = [], []
    for x in X:

        net.wf.rbf.centers.data[0,2] = -x
        net.wf.rbf.centers.data[0,2] = x
        pos = Variable(net.sample())
        pos.requires_grad = True
        e = net.wf.energy(pos)
        s = net.wf.variance(pos)

        energy.append(e)
        var.append(s)

    plt.plot(X,energy)
    plt.show()
    plt.plot(X,var)
    plt.show()    

if 0:
    net.wf.layer_mo.weight.data = torch.eye(net.wf.nao).double()
    for param in net.wf.layer_ci.parameters():
        param.requires_grad = False


    pos = net.sample(ntherm=0)
    pos = pos.reshape(100,100,6)
    var_ = net.observalbe(net.wf.variance,pos)
    plt.plot(var_)
    plt.show()

    net.train(50,pos=pos,ntherm=-1)