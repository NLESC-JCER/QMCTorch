import adaptive
from adaptive.learner.learner1D import (curvature_loss_function,
                                        uniform_loss,
                                        default_loss)

import torch
import numpy as np 

def torchify(func):
    '''Transform a torch funtion to accept non torch argument.

    Example :

    def func(pos):
        # pos needs here to be a torch tensor
        return torch.exp(-pos)

    newf = torchify(func)
    newf(1.)
    '''

    #assert func.__code__.co_argcount == 1
    def torchf(x):
        x = torch.tensor(float(x))
        data = func(x).detach().numpy().flatten().tolist()
        if len(data) == 1:
            data = data[0]
        return data
    return torchf

def fgradient(func,eps=1E-6):
    def _grad(x):
        return (func(x+eps)+func(x-eps)-2*func(x))/eps**2
    return _grad

def finverse(func,eps=1E-12):
    def _inv(x):
        return 1./(func(x)+eps)
    return _inv

def amplitude_1d(xs,ys):
    from math import sqrt
    return (xs[1]-xs[0])/sqrt((ys[1]-ys[0])**2+1)

def adaptive_mesh_1d(func,xmin,xmax,npoints,loss='default'):

	losses = {'amplitude' : amplitude_1d,
			  'uniform'   : uniform_loss,
              'default'   : default_loss,
			  'curvature' : curvature_loss_function()}

	learner = adaptive.Learner1D(func, (xmin, xmax), loss_per_interval=losses[loss])
	npoints_goal = lambda l: l.npoints >= npoints
	adaptive.runner.simple(learner, goal=npoints_goal)
	
	points = list(learner.data.keys())

	return points


def adaptive_mesh_2d(func,xmin,xmax,ymin,ymax,nx,ny,loss='amplitude'):

    losses = {'amplitude' : amplitude_1d,
              'uniform'   : uniform_loss,
              'default'   : default_loss,
              'curvature' : curvature_loss_function()}

    learner = adaptive.Learner1D(func, (xmin, xmax), loss_per_interval=losses[loss])
    npoints_goal = lambda l: l.npoints >= npoints
    adaptive.runner.simple(learner, goal=npoints_goal)
    
    points = list(learner.data.keys())

    return points

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