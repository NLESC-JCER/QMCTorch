import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import torch

from .stat_utils import (blocking, correlation_coefficient,
                         integrated_autocorrelation_time,
                         fit_correlation_coefficient)


def Display_orbital(compute_mo, wf, plane="z", plane_coord = 0.00, 
                start = -5, end = 5, step = 0.1, path=None, 
                title=None, orbital_ind = [0,0], spin = "up"):
    """"Function in attempt to visualise the orbital behaviour of the FermiNet.
        The function will display the first orbital of the first determinant.
        All electrons except one will be kept at a constant position.
        The output of the wave function will be determined over a 2D grid on a given plane.

        Args:
            compute_mo (compute mo, method): method that computes the MOs from the pos.
            plane (str, optional): The axis orthogonal to the grid plane.
                                    Default to "z": options ["x","y","z"]
            plane_coord (float, optional): Constant coordinate on the plane axis.
                                    Default to 0.00
            start (float, optional): Starting grid point. Default: -5
            end (float, optional): End grid point. Default: 5
            step (float, optional): step size of grid. Default: 0.1
            path (str, optional): path/filename to save the plot to   
            title (str, optional): title of the plot  
            orbital_ind (tupple, optional): which orbital of the determinant to plot [det, orbital]
    """
    # keep all electrons except one at a constant position to:
    dim = ["x","y","z"]
    if plane not in dim: 
        ValueError("{} is not a valid plane. choose from {}.".format(plane,dim))
    plane_index = dim.index(plane)
    index =[0,1,2]
    index.pop(plane_index)

    det, orbital = orbital_ind

    grid = torch.arange(start ,end,
                        step, device="cpu")

    pos_1 = torch.zeros(len(grid), len(grid), 3)


    grid1, grid2 = torch.meshgrid(grid,grid)
    grid3 = plane_coord*torch.ones((grid1.shape[0],grid1.shape[1]))   
    grid12 = torch.cat((grid1.unsqueeze(2),grid2.unsqueeze(2)),dim=2)
    pos_1[:,:,index] = grid12
    pos_1[:,:,plane_index] = grid3
    pos_1 = pos_1.reshape(grid1.shape[0]**2,3)

    # which electron to move:
    if spin == "up":
        elec_ind = 0
    elif spin == "down":
        elec_ind = wf.mol.nup
    else :
        ValueError("{} is not a valid spin. spin is either up or down.".format(spin)) 
    
    # all other electrons at constant position (1,1,1)
    pos = torch.ones((pos_1.shape[0],wf.mol.nelec,wf.ndim), device="cpu")  
    pos[:,elec_ind] = pos_1
    
    pos = pos.reshape((pos.shape[0],wf.mol.nelec*wf.ndim))

    mo_up, mo_down = compute_mo(pos)
    if spin == "up":
        mo = mo_up
    else:
        mo = mo_down
   
    mo = mo.detach().reshape((grid1.shape[0],
                        grid1.shape[0],mo.shape[1],
                        mo.shape[2], mo.shape[3]))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(grid1.numpy(), grid2.numpy(), mo[:,:,det,orbital,0].numpy())
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(dim[index[0]])
    ax.set_ylabel(dim[index[1]])
    ax.set_zlabel(r'$\phi({},{},{}={})$'.format(dim[index[0]], dim[index[1]], plane, plane_coord))

    if path is not None:
        fig.savefig(path)
    else :
        fig.show()
