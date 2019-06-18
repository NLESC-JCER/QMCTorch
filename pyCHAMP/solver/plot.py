import torch
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure

from pyCHAMP.solver.mesh import regular_mesh_2d, regular_mesh_3d


def plot_observable(obs_dict,e0=None,ax=None):
    '''Plot the observable selected.

    Args:
        obs_dict : dictioanry of observable
    '''
    show_plot = False
    if ax is None:    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        show_plot = True

    n = len(obs_dict['energy'])
    epoch = np.arange(n)

    emax = [np.quantile(e,0.75) for e in obs_dict['local_energy'] ]
    emin = [np.quantile(e,0.25) for e in obs_dict['local_energy'] ]

    ax.fill_between(epoch,emin,emax,alpha=0.5,color='#4298f4')
    ax.plot(epoch,obs_dict['energy'],color='#144477')
    if e0 is not None:
        ax.axhline(e0,color='black',linestyle='--')

    ax.grid()
    ax.set_xlabel('Number of epoch')
    ax.set_ylabel('Energy')

    if show_plot:
        plt.show()


###########################################################################################
##  1D routnines
###########################################################################################

class plotter1d(object):

    def __init__(self, wf, domain, res, sol = None):
        '''Dynamic plot of a 1D-wave function during the optimization

        Args:
            wf : wave function object
            domain : dict containing the boundary
            res : number of points in each direction
            sol : callabale solution of the problem
        '''

        self.wf = wf
        self.res = res
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111 )

        self.POS = Variable(torch.linspace(domain['xmin'],domain['xmax'],res).view(res,1))
        pos = self.POS.detach().numpy().flatten()  

        if callable(sol):
            self.ax.plot(pos,sol(pos),color='blue')

        vp = self.wf(self.POS).detach().numpy()
        print(pos.shape)
        print(vp.shape)
        self.lwf, = self.ax.plot(pos,vp,color='red')
        self.pweight, = self.ax.plot(self.wf.rbf.centers.detach().numpy(),self.wf.fc.weight.detach().numpy().T,'o')
        self.pgrad, = self.ax.plot(self.wf.rbf.centers.detach().numpy(),np.zeros(self.wf.ncenter),'X')

        plt.draw()
        self.fig.canvas.flush_events()

    def drawNow(self):
        '''Update the plot.'''

        vp = self.wf(self.POS).detach().numpy()
        self.lwf.set_ydata(vp)

        self.pweight.set_xdata(self.wf.rbf.centers.detach().numpy())
        self.pweight.set_ydata(self.wf.fc.weight.detach().numpy().T)

        self.pgrad.set_xdata(self.wf.rbf.centers.detach().numpy())
        data = (self.wf.fc.weight.grad.detach().numpy().T)**2
        data /= np.linalg.norm(data)
        self.pgrad.set_ydata(data)

        self.fig.canvas.draw()  

def plot_wf_1d(net,grad=False,hist=False,sol=None):
        '''Plot a 1D wave function.

        Args:
            net : network object
            grad : plot gradient
            hist : plot histogram of the data points
            sol : callabale of the solution
        '''

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

def plot_results_1d(net,obs_dict,sol=None,e0=None,xmin=-5,xmax=5,nx=100):
    ''' Plot the summary of the results for a 1D problem.

    Args: 
        net : network object
        obs_dict : dict containing the obserable
        sol : callable of the solutions
        e0 : energy of the solution
        domain : boundary of the plot
        res : number of points in the x axis
    '''

    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    X = Variable(torch.linspace(xmin,xmax,nx).view(nx,1))
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

###########################################################################################
##  2D routnines
###########################################################################################

def plot_wf_2d(net,domain,res,sol=None):
        '''Plot a 2D wave function.

        Args:
            net : network object
            grad : plot gradient
            hist : plot histogram of the data points
            sol : callabale of the solution
        '''

        fig = plt.figure()
        ax = fig.add_subplot( 111, projection='3d' )

        points = regular_mesh_2d(xmin=domain['xmin'],xmax=domain['xmax'],
                                ymin=domain['ymin'],ymax=domain['ymax'],
                                nx=res[0],ny=res[1])

        POS = Variable(torch.tensor(points))
        POS.requires_grad = True

        pos = POS.detach().numpy()
        xx = pos[:,0].reshape(res[0],res[1])
        yy = pos[:,1].reshape(res[0],res[1])

        if callable(sol):
            vs = sol(POS).view(res[0],res[1]).detach().numpy()
            vs /= np.linalg.norm(vs)
            ax.plot_wireframe(xx,yy,vs,color='black',linewidth=1)

        vals = net.wf(POS)
        vn = vals.detach().numpy().reshape(res[0],res[1])
        vn /= np.linalg.norm(vn)
        ax.plot_surface(xx,yy,vn,cmap=cm.coolwarm,alpha=0.75,color='black',linewidth=2)
    
        plt.show()

class plotter2d(object):

    def __init__( self, wf, domain, res, pot=False, kinetic=False, sol=None ):

        '''Dynamic plot of a 2D-wave function during the optimization

        Args:
            wf : wave function object
            domain : dict containing the boundary
            res : number of points in each direction
            sol : callabale solution of the problem
        '''

        self.wf = wf
        self.res = res
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111, projection='3d' )

        points = regular_mesh_2d(xmin=domain['xmin'],xmax=domain['xmax'],
                                 ymin=domain['ymin'],ymax=domain['ymax'],
                                 nx=res[0],ny=res[1])

        self.POS = Variable(torch.tensor(points))
        self.POS.requires_grad = True

        pos = self.POS.detach().numpy()
        self.xx = pos[:,0].reshape(res[0],res[1])
        self.yy = pos[:,1].reshape(res[0],res[1])


        if callable(sol):
            vs = sol(self.POS).view(self.res[0],self.res[1]).detach().numpy()
            vs /= np.linalg.norm(vs)
            self.ax.plot_wireframe(self.xx,self.yy,vs,color='black',linewidth=1)

        if pot:
            vs = wf.nuclear_potential(self.POS).view(self.res[0],self.res[1]).detach().numpy()
            self.ax.plot_wireframe(self.xx,self.yy,vs,color='red',linewidth=1)


        if kinetic:
            vs = wf.kinetic_energy(self.POS).view(self.res[0],self.res[1]).detach().numpy()
            self.ax.plot_wireframe(self.xx,self.yy,10*vs,color='red',linewidth=1)

        self.vals = self.wf(self.POS).view(self.res[0],self.res[1]).detach().numpy()

        self.surf = self.ax.plot_surface( 
                        self.xx, self.yy, self.vals, rstride=1, cstride=1, 
                        cmap=cm.coolwarm,alpha=0.75,color='black',linewidth=2,
                        antialiased=False )
        plt.draw()
        self.fig.canvas.flush_events()

    def drawNow( self ):
        '''update the plot.'''
        self.surf.remove()
        self.vals = self.wf(self.POS).view(self.res[0],self.res[1]).detach().numpy()
        self.surf = self.ax.plot_surface( 
                        self.xx, self.yy, self.vals, rstride=1, cstride=1, 
                        cmap=cm.coolwarm,alpha=0.75,color='black',linewidth=2,
                        antialiased=False )
        plt.draw()                     
        self.fig.canvas.flush_events()

def plot_results_2d(net,obs_dict,domain,res,sol=None,e0=None):
    ''' Plot the summary of the results for a 1D problem.

    Args: 
        net : network object
        obs_dict : dict containing the obserable
        domain : boundary of the plot
        res : number of points in the x axis
        sol : callable of the solutions
        e0 : energy of the solution
    '''

    fig = plt.figure()
    ax0 = fig.add_subplot(211, projection='3d')
    ax1 = fig.add_subplot(212)

    points = regular_mesh_2d(xmin=domain['xmin'],xmax=domain['xmax'],
                            ymin=domain['ymin'],ymax=domain['ymax'],
                            nx=res[0],ny=res[1])

    POS = Variable(torch.tensor(points))
    POS.requires_grad = True

    pos = POS.detach().numpy()
    xx = pos[:,0].reshape(res[0],res[1])
    yy = pos[:,1].reshape(res[0],res[1])

    if callable(sol):
        vs = sol(POS).view(res[0],res[1]).detach().numpy()
        vs /= np.linalg.norm(vs)
        ax0.plot_wireframe(xx,yy,vs,color='black',linewidth=1)

    vals = net.wf(POS)
    vn = vals.detach().numpy().reshape(res[0],res[1])
    vn /= np.linalg.norm(vn)
    ax0.plot_surface(xx,yy,vn,cmap=cm.coolwarm,alpha=0.75,color='black',linewidth=2)

    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.set_zlabel('Wavefunction')

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

###########################################################################################
##  3D routnines
###########################################################################################

def plot_results_3d(net,obs_dict,domain,res,isoval=0.02,sol=None,e0=None,hist=False):
    ''' Plot the summary of the results for a 1D problem.

    Args: 
        net : network object
        obs_dict : dict containing the obserable
        domain : boundary of the plot
        res : number of points in the x axis
        sol : callable of the solutions
        e0 : energy of the solution
    '''
    
    fig = plt.figure()
    ax0 = fig.add_subplot(211, projection='3d')
    ax1 = fig.add_subplot(212)

    plot_wf_3d(net,domain,res,sol=sol,isoval=isoval,hist=hist,ax=ax0)
    plot_observable(obs_dict,e0=e0,ax=ax1)

    plt.show()

def plot_wf_3d(net,domain,res,sol=None,
               wf=True, isoval=0.02,
               pot=False,pot_isoval=0,
               grad=False,grad_isoval=0,
               hist=False,ax=None):
    '''Plot a 3D wave function.

    Args:
        net : network object
        grad : plot gradient
        hist : plot histogram of the data points
        sol : callabale of the solution
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot( 111, projection='3d' )
        show_plot = True
    else:
        show_plot = False

    points = regular_mesh_3d(xmin=domain['xmin'],xmax=domain['xmax'],
                            ymin=domain['ymin'],ymax=domain['ymax'],
                            zmin=domain['zmin'],zmax=domain['zmax'],
                            nx=res[0],ny=res[1],nz=res[2])

    POS = Variable(torch.tensor(points))
    POS.requires_grad = True

    pos = POS.detach().numpy()
    xx = pos[:,0].reshape(res[0],res[1],res[2])
    yy = pos[:,1].reshape(res[0],res[1],res[2])
    zz = pos[:,2].reshape(res[0],res[1],res[2])


    dx = (domain['xmax']-domain['xmin'])/ (res[0]-1)
    dy = (domain['ymax']-domain['ymin'])/ (res[1]-1)
    dz = (domain['zmax']-domain['zmin'])/ (res[2]-1)
    spacing_vals = (dx,dy,dz)

    if hist:
        pos = net.sample().detach().numpy()
        for ielec in range(net.wf.nelec):
            ax.scatter(pos[:,ielec*3],pos[:,ielec*3+1],pos[:,ielec*3+2])

    if callable(sol):

        vals = sol(POS)
        vs = vals.detach().numpy().reshape(res[0],res[1],res[2])
        verts, faces, normals,values = measure.marching_cubes_lewiner(vs,isoval,spacing=spacing_vals)

        ax.plot_trisurf(verts[:,0]+domain['xmin'],
                        verts[:,1]+domain['ymin'],
                        faces,
                        verts[:,2]+domain['zmin'],
                        alpha=0.25,antialiased=True,
                        lw=1, edgecolor='blue')

    if pot:

        vals = net.wf.nuclear_potential(POS)
        vn = vals.detach().numpy().reshape(res[0],res[1],res[2])
        verts, faces, normals,values = measure.marching_cubes_lewiner(vn,pot_isoval,spacing=spacing_vals)        
        
        ax.plot_trisurf(verts[:,0]+domain['xmin'],
                        verts[:,1]+domain['ymin'],
                        faces,
                        verts[:,2]+domain['zmin'],
                        alpha=0.25,antialiased=True,
                        lw=1, edgecolor='red')

    if grad:

        #vals = net.wf.kinetic_energy_finite_difference(POS,eps=1E-3)
        vals = net.wf.kinetic_energy(POS)
        vn = vals.detach().numpy().reshape(res[0],res[1],res[2])
        verts, faces, normals,values = measure.marching_cubes_lewiner(vn,grad_isoval,spacing=spacing_vals)        
        
        ax.plot_trisurf(verts[:,0]+domain['xmin'],
                        verts[:,1]+domain['ymin'],
                        faces,
                        verts[:,2]+domain['zmin'],
                        alpha=0.25,antialiased=True,
                        lw=1, edgecolor='green')
    if wf:

        vals = net.wf(POS)
        vn = vals.detach().numpy().reshape(res[0],res[1],res[2])
        verts, faces, normals,values = measure.marching_cubes_lewiner(vn,isoval,spacing=spacing_vals)        
            
        ax.plot_trisurf(verts[:,0]+domain['xmin'],
                        verts[:,1]+domain['ymin'],
                        faces,
                        verts[:,2]+domain['zmin'],
                        alpha=0.5,cmap='Spectral',
                        antialiased=True,lw=1,edgecolor='black')

    ax.set_xlim(domain['xmin'],domain['xmax'])
    ax.set_ylim(domain['ymin'],domain['ymax'])
    ax.set_zlim(domain['zmin'],domain['zmax'])

    if show_plot:
        plt.show()
