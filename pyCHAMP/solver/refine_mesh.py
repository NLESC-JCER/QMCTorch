import torch
import numpy as np 


def refine_mesh(wf, qmin=0.25, qmax = 0.75):

    if wf.ndim == 1:
        return refine_mesh_1d(wf, qmin=qmin, qmax = qmax) 
    else:
        raise ValueError('Refinement only possible for 1D grid')

def refine_mesh_1d(wf, qmin=0.25, qmax = 0.75):
    
    '''Refine a 1D mesh by coarsening/refining the grid.

    Args:
        wf : wavefunction 
        qmin : lower limit for coarsening
        qmax : limit for refininf
    '''

    # get the values of thw wave function and the limits
    vals = wf(wf.centers).detach().numpy()
    vmin = np.quantile(vals,qmin)
    vmax = np.quantile(vals,qmax)

    # get boolean on point data
    bool_min = vals < vmin
    bool_max = vals > vmax

    # get boolean on interval data
    interval_min = np.logical_and(bool_min[:-1],bool_min[1:])
    interval_max = np.logical_and(bool_max[:-1],bool_max[1:])
    ninterval = len(interval_min)
    
    # copy old values in numpy arrays
    old_centers = wf.centers.detach().numpy().flatten()
    old_fc_weight = wf.fc.weight.data.detach().numpy().flatten()

    # init the new arrays
    new_centers = []
    new_fc_weight = []

    # loop over the intervals
    i = 0
    while i < ninterval-1:

        if interval_min[i] and interval_min[i+1]:

            new_centers.append(np.mean(old_centers[i:i+2]))
            new_fc_weight.append(np.mean(old_fc_weight[i:i+2]))

            new_centers.append(np.mean(old_centers[i+1:i+3]))
            new_fc_weight.append(np.mean(old_fc_weight[i+1:i+3]))
            i += 2

        # refine the grid
        elif interval_max[i]:

            new_centers.append(old_centers[i])
            new_fc_weight.append(old_fc_weight[i])

            new_centers.append(np.mean(old_centers[i:i+2]))
            new_fc_weight.append(np.mean(old_fc_weight[i:i+2]))
            i += 1

        # keep the grid
        else:

            new_centers.append(old_centers[i])
            new_fc_weight.append(old_fc_weight[i])
            i += 1

    # append the last point if necessary
    if not interval_min[-1]:
        
        new_centers.append(old_centers[i])
        new_fc_weight.append(old_fc_weight[i])

    return new_centers, new_fc_weight



