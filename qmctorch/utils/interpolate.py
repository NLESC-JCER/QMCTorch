import torch
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def get_grid(atomic_positions, resolution=0.5, border_length=2,):
    """Computes grid points frm the atomic positions

    Args:
        atomic_positions (torch.Tensor, np.ndarray, list): atomic positions
        resolution (float, optional): ditance between two points. Defaults to 0.5.
        border_length (float, optional): length of the border. Defaults to 2.

    Raises:
        ValueError: if type of positions not recognized

    Returns:
        (np.ndarray, np.ndarray, mp.ndarray): grid points in the x, y and z axis
    """

    if isinstance(atomic_positions, torch.Tensor):
        pmin = atomic_positions.min(0)[0].detach().cpu().numpy()
        pmax = atomic_positions.max(0)[0].detach().cpu().numpy()

    elif isinstance(atomic_positions, np.ndarray):
        pmin, pmax = atomic_positions.min(0), atomic_positions.max(0)

    elif isinstance(atomic_positions, list):
        _tmp = np.array(atomic_positions)
        pmin, pmax = _tmp.min(0), _tmp.max(0)

    else:
        raise ValueError(
            'atomic_positions must be either a torch.tensor, np.ndarray, or list')

    pmin -= border_length
    pmax += border_length
    npts = np.ceil((pmax-pmin) / resolution).astype('int')

    x = np.linspace(pmin[0], pmax[0], npts[0])
    y = np.linspace(pmin[1], pmax[1], npts[1])
    z = np.linspace(pmin[2], pmax[2], npts[2])

    return (x, y, z)


def interpolator_regular_grid(func, x, y, z):
    """Computes the interpolation function

    Args:
        func (callable):  compute the value of the funtion to interpolate
        x (np.ndarray): grid points in the x direction
        y (np.ndarray): grid points in the y direction
        z (np.ndarray): grid points in the z direction

    Returns:
        callable: interpolation function
    """
    nx, ny, nz = len(x), len(y), len(z)
    grid = np.stack(np.meshgrid(
        x, y, z, indexing='ij')).T.reshape(-1, 3)
    grid = torch.tensor(grid)
    data = func(grid)
    data = data.transpose(0, 1).reshape(nx, ny, nz, -1)
    return RegularGridInterpolator((x, y, z), data.detach().numpy())


def interpolate_regular_grid(interpfunc, pos):
    """Interpolate the  funtion

    Args:
        interpfunc (callable): function to interpolate the data points
        pos (torch.tensor): positions of the walkers Nbatch x 3*Nelec

    Returns:
        torch.tensor: interpolated values of the function evaluated at pos
    """
    nbatch = pos.shape[0]
    nelec = pos.shape[1]//3
    ndim = 3
    data = interpfunc(pos.reshape(
        nbatch, nelec, ndim).transpose(0, 1))

    return torch.tensor(data.transpose(1, 0, 2))
