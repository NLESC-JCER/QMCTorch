import torch
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from deepqmc.solver.plot_data import plot_observable


def regular_mesh_2d(xmin=-2,xmax=2,ymin=-2.,ymax=2,nx=5,ny=5):

    x = np.linspace(xmin,xmax,nx)
    y = np.linspace(ymin,ymax,ny)

    XX,YY = np.meshgrid(x,y)
    points = np.vstack( (XX.flatten(),YY.flatten()) ).T

    return points.tolist()

def regular_mesh_3d(xmin=-2,xmax=2,ymin=-2.,ymax=2,zmin=-5,zmax=5,nx=5,ny=5,nz=5):

    x = np.linspace(xmin,xmax,nx)
    y = np.linspace(ymin,ymax,ny)
    z = np.linspace(zmin,zmax,nz)

    XX,YY,ZZ = np.meshgrid(x,y,z)
    points = np.vstack( (XX.flatten(),YY.flatten(),ZZ.flatten()) ).T

    return points.tolist()


###########################################################################################
##  1D routines
###########################################################################################

class plotter1d(object):

    def __init__(self, wf, domain, res=51, sol = None, plot_weight=False, plot_grad=False):
        '''Dynamic plot of a 1D-wave function during the optimization

        Args:
            wf : wave function object
            domain : dict containing the boundary
            res : number of points in each direction
            sol : callabale solution of the problem
            plot_weight : plot the weight of the fc
            plot_grad : plot the grad of the weight
        '''
        plt.ion()
        self.wf = wf
        self.res = res
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111 )

        self.plot_weight = plot_weight
        self.plot_grad = plot_grad

        self.POS = Variable(torch.linspace(domain['xmin'],domain['xmax'],res).view(res,1))
        pos = self.POS.detach().numpy().flatten()  

        if callable(sol):
            v = sol(self.POS).detach().numpy()
            self.ax.plot(pos,v,color='#b70000',linewidth=4,linestyle='--',label='solution')

        vpot = wf.nuclear_potential(self.POS).detach().numpy()
        self.ax.plot(pos,vpot,color='black',linestyle='--')

        vp = self.wf(self.POS).detach().numpy()
        vp/=np.max(vp)
        self.lwf, = self.ax.plot(pos,vp,linewidth=2,color='black')

        if self.plot_weight:
            self.pweight, = self.ax.plot(self.wf.rbf.centers.detach().numpy(),
                                         self.wf.fc.weight.detach().numpy().T,'o')
        if self.plot_grad:
            if self.wf.fc.weight.requires_grad:
                self.pgrad, = self.ax.plot(self.wf.rbf.centers.detach().numpy(),
                                           np.zeros(self.wf.ncenter),'X')

        self.ax.set_ylim((np.min(vpot),1))
        plt.grid()
        plt.draw()
        self.fig.canvas.flush_events()

    def drawNow(self):
        '''Update the plot.'''

        vp = self.wf(self.POS).detach().numpy()
        vp/=np.max(vp)
        self.lwf.set_ydata(vp)

        if self.plot_weight:
            self.pweight.set_xdata(self.wf.rbf.centers.detach().numpy())
            self.pweight.set_ydata(self.wf.fc.weight.detach().numpy().T)

        if self.plot_grad:
            if self.wf.fc.weight.requires_grad:
                self.pgrad.set_xdata(self.wf.rbf.centers.detach().numpy())
                data = (self.wf.fc.weight.grad.detach().numpy().T)**2
                data /= np.linalg.norm(data)
                self.pgrad.set_ydata(data)

        #self.fig.canvas.draw() 
        plt.draw()
        self.fig.canvas.flush_events()

def plot_wf_1d(net,domain,res,grad=False,hist=False,pot=True,sol=None,ax=None,load=None):
        '''Plot a 1D wave function.

        Args:
            net : network object
            grad : plot gradient
            hist : plot histogram of the data points
            sol : callabale of the solution
        '''

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot( 111 )
            show_plot = True
        else:
            show_plot = False


        if load is not None:
            checkpoint = torch.load(load)
            net.wf.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']

        X = Variable(torch.linspace(domain['xmin'],domain['xmax'],res).view(res,1))
        X.requires_grad = True
        xn = X.detach().numpy().flatten()

        if callable(sol):
            vs = sol(X).detach().numpy()
            ax.plot(xn,vs,color='#b70000',linewidth=4,linestyle='--',label='solution')

        vals = net.wf(X)
        vn = vals.detach().numpy().flatten()
        vn /= np.max(vn)
        ax.plot(xn,vn,color='black',linewidth=2,label='DeepQMC')

        if pot:
            pot = net.wf.nuclear_potential(X).detach().numpy()
            ax.plot(xn,pot,color='black',linestyle='--')

        if grad:
            kin = net.wf.kinetic_energy(X)
            g = np.gradient(vn,xn)
            h = -0.5*np.gradient(g,xn)
            ax.plot(xn,kin.detach().numpy(),label='kinetic')
            ax.plot(xn,h,label='hessian')

        if hist:
            pos = net.sample(ntherm=-1)
            ax.hist(pos.detach().numpy(),density=False)
        
        ax.set_ylim((np.min(pot),1))
        ax.grid()
        ax.set_xlabel('X')
        if load is None:
            ax.set_ylabel('Wavefuntion')
        else:
            ax.set_ylabel('Wavefuntion %d epoch' %epoch)
        ax.legend()

        if show_plot:
            plt.show()

def plot_results_1d(net,domain,res,sol=None,e0=None,load=None):
    ''' Plot the summary of the results for a 1D problem.

    Args: 
        net : network object
        obs_dict : dict containing the obserable
        sol : callable of the solutions
        e0 : energy of the solution
        domain : boundary of the plot
        res : number of points in the x axis
    '''
    plt.ioff()
    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    plot_wf_1d(net,domain,res,sol=sol,hist=False,ax=ax0,load=load)
    plot_observable(net.obs_dict,e0=e0,ax=ax1)

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
        plt.ion()

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
    plt.ioff()
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

def plot_results_3d(net,obs_dict,domain,res,wf=False,isoval=0.02,sol=None,e0=None,hist=False):
    ''' Plot the summary of the results for a 1D problem.

    Args: 
        net : network object
        obs_dict : dict containing the obserable
        domain : boundary of the plot
        res : number of points in the x axis
        sol : callable of the solutions
        e0 : energy of the solution
    '''
    plt.ioff()
    fig = plt.figure()
    ax0 = fig.add_subplot(211, projection='3d')
    ax1 = fig.add_subplot(212)

    plot_wf_3d(net,domain,res,wf=wf,sol=sol,isoval=isoval,hist=hist,ax=ax0)
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



