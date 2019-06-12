import adaptive
from adaptive.learner.learner1D import (curvature_loss_function,
                                        uniform_loss,
                                        default_loss)

import torch
import numpy as np 

def torchify(func):
    '''Transform a torch funtion to accept non torch argument.

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

def amplitude_1d(xs,ys):
    from math import sqrt
    return (xs[1]-xs[0])/sqrt((ys[1]-ys[0])**2+1)

def adaptive_mesh_1d(func,xmin,xmax,npoints,loss='amplitude'):

	losses = {'amplitude' : amplitude_1d,
			  'uniform'   : uniform_loss,
			  'curvature'  : curvature_loss_function()}

	learner = adaptive.Learner1D(func, (xmin, xmax), loss_per_interval=losses[loss])
	npoints_goal = lambda l: l.npoints >= npoints
	adaptive.runner.simple(learner, goal=npoints_goal)
	
	points = list(learner.data.keys())

	return points