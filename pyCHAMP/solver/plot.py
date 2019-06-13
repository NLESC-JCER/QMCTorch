import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def plot_observable(obs_dict):

    n = len(obs_dict['energy'])
    epoch = np.arange(n)

    emax = [np.quantile(e,0.75) for e in obs_dict['local_energy'] ]
    emin = [np.quantile(e,0.25) for e in obs_dict['local_energy'] ]

    plt.fill_between(epoch,emin,emax,alpha=0.5,color='#4298f4')
    plt.plot(epoch,obs_dict['energy'],color='#144477')
    plt.grid()
    plt.xlabel('Number of epoch')
    plt.ylabel('Energy')
    plt.show()

def plot_wf_1d(net,grad=False,hist=False,sol=None):

        X = Variable(torch.linspace(-5,5,100).view(100,1,1))
        X.requires_grad = True
        xn = X.detach().numpy().flatten()

        if callable(sol):
            vs = sol(xn)
            plt.plot(xn,vs,color='#b70000',linewidth=4,linestyle='--')

        vals = net.wf(X)
        vn = vals.detach().numpy().flatten()
        vn /= np.linalg.norm(vn)
        plt.plot(xn,vn,color='black',linewidth=2)

        if pot:
            pot = net.wf.nuclear_potential(X).detach().numpy()
            plt.plot(xn,pot,color='black',linestyle='--')

        if grad:
            kin = net.wf.kinetic_autograd(X)
            g = np.gradient(vn,xn)
            h = np.gradient(g,xn)
            plt.plot(xn,kin.detach().numpy())
            plt.plot(xn,h)

        if hist:
            pos = net.sample(ntherm=-1)
            plt.hist(pos.detach().numpy(),density=False)
        
        plt.grid()
        plt.show()

def plot_results_1d(net,obs_dict,sol=None,e0=None,xmin=-5,xmax=-5,nx=100):

    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    X = Variable(torch.linspace(xmin,xmax,nx).view(nx,1,1))
    X.requires_grad = True
    xn = X.detach().numpy().flatten()

    if callable(sol):
        vs = sol(xn)
        ax0.plot(xn,vs,color='#b70000',linewidth=4,linestyle='--',label='Solution')

    vals = net.wf(X)
    vn = vals.detach().numpy().flatten()
    vn /= np.linalg.norm(vn)
    ax0.plot(xn,vn,color='black',linewidth=2,label='DeepQMC')

    pot = net.wf.nuclear_potential(X).detach().numpy()
    ax0.plot(xn,pot/10,color='black',linestyle='--',label='V(x)')

    ax0.set_ylim((np.min(pot/10),np.max(vn)))
    ax0.grid()
    ax0.set_xlabel('X')
    ax0.set_ylabel('Wavefuntion')
    ax0.legend()

    n = len(obs_dict['energy'])
    epoch = np.arange(n)

    emax = [np.quantile(e,0.75) for e in obs_dict['local_energy'] ]
    emin = [np.quantile(e,0.25) for e in obs_dict['local_energy'] ]

    ax1.fill_between(epoch,emin,emax,alpha=0.5,color='#4298f4')
    ax1.plot(epoch,obs_dict['energy'],color='#144477')
    if e0 is not None:
        ax1.axhline(e0,color='black',linestyle='--')

    ax1.grid()
    ax1.set_xlabel('Number of epoch')
    ax1.set_ylabel('Energy')

    plt.show()


def plot_wf_2d(net,sol=None,xmin=-5,xmax=5,nx=5,ymin=-5,ymax=5,ny=5):

        from pyCHAMP.solver.mesh import regular_mesh_2d as mesh

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

        points = mesh(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,nx=nx,ny=ny)

        POS = Variable(torch.tensor(points))
        POS.requires_grad = True

        pos = POS.detach().numpy()
        xx = pos[:,0].reshape(nx,ny)
        yy = pos[:,1].reshape(nx,ny)

        if callable(sol):
            vs = sol(POS).view(nx,ny).detach().numpy()
            ax.plot_wireframe(xx,yy,vs,color='black',linewidth=1)

        vals = net.wf(POS)
        vn = vals.detach().numpy().reshape(nx,ny)
        vn /= np.linalg.norm(vn)
        ax.plot_surface(xx,yy,vn,cmap=cm.coolwarm,alpha=0.5,color='black',linewidth=2)
    
        plt.show()