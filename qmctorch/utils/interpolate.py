import torch
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator


def get_boundaries(atomic_positions, border_length=2.):
    """Computes the boundaries of the structure

    Args:
        atomic_positions (torch.Tensor, np.ndarray, list): atomic positions
        border_length (float, optional): length of the border. Defaults to 2.

    Raises:
        ValueError: if type of positions not recognized

    Returns:
        (np.ndarray, np.ndarray, mp.ndarray): min, max values in the 3 cartesian directions
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

    return pmin, pmax


def get_reg_grid(atomic_positions, resolution=0.1, border_length=2.):
    """Computes a regular grid points from the atomic positions

    Args:
        atomic_positions (torch.Tensor, np.ndarray, list): atomic positions
        resolution (float, optional): ditance between two points. Defaults to 0.5.
        border_length (float, optional): length of the border. Defaults to 2.

    Returns:
        (np.ndarray, np.ndarray, mp.ndarray): grid points in the x, y and z axis
    """

    pmin, pmax = get_boundaries(
        atomic_positions, border_length=border_length)
    npts = np.ceil((pmax-pmin) / resolution).astype('int')

    x = np.linspace(pmin[0], pmax[0], npts[0])
    y = np.linspace(pmin[1], pmax[1], npts[1])
    z = np.linspace(pmin[2], pmax[2], npts[2])

    return (x, y, z)


def interpolator_reg_grid(func, x, y, z):
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
        z, y, x, indexing='ij')).T.reshape(-1, 3)[:, [2, 1, 0]]
    grid = torch.tensor(grid)
    data = func(grid).detach().numpy()
    data = data.reshape(nx, ny, nz, -1)
    return RegularGridInterpolator((x, y, z),
                                   data,
                                   method='linear',
                                   bounds_error=False,
                                   fill_value=0.)


def interpolate_reg_grid(interpfunc, pos):
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
        nbatch, nelec, ndim).detach().numpy())

    return torch.tensor(data)


def is_even(x):
    """return true if x is even."""
    return x//2*2 == x


def logspace(n, length):
    """returns a 1d array of logspace between -length and +length."""
    k = np.log(length+1)/np.log(10)
    if is_even(n):
        x = np.logspace(0.01, k, n//2)-1
        return np.concatenate((-x[::-1], x[1:]))
    else:
        x = np.logspace(0.0, k, n//2+1)-1
        return np.concatenate((-x[::-1], x[1:]))


def get_log_grid(atomic_positions, n=6, length=2., border_length=2.):
    """Computes a logarithmic grid 

    Args:
        atomic_positions (list, np.ndarray, torch.tensor): positions of the atoms
        n (int, optional): number of points in each axis around each atom. Defaults to 6.
        length (float, optional): absolute value of the max distance from the atom. Defaults to 2.
        border_length (float, optional): length of the border. Defaults to 2.

    Returns:
        np.ndanrray: grid points (Npts,3)
    """

    x, y, z = np.stack(get_boundaries(
        atomic_positions, border_length=border_length)).T
    grid_pts = np.stack(np.meshgrid(
        x, y, z, indexing='ij')).T.reshape(-1, 3)

    x = logspace(n, length)
    pts = np.stack(np.meshgrid(x, x, x, indexing='ij')
                   ).T.reshape(-1, 3)

    for pos in atomic_positions:
        _tmp = pts + pos
        if grid_pts is None:
            grid_pts = _tmp
        else:
            grid_pts = np.concatenate((grid_pts, _tmp))
    return grid_pts


def interpolator_irreg_grid(func, grid_pts):
    """compute a linear ND interpolator

    Args:
        func (callable):  compute the value of the funtion to interpolate
        grid_pts (np.ndarray): grid points in the x direction

    Returns:
        callable: interpolation function
    """

    return LinearNDInterpolator(grid_pts,
                                func(grid_pts),
                                fill_value=0.)


def interpolate_irreg_grid(interpfunc, pos):
    """Interpolate the  funtion

    Args:
        interpfunc (callable): function to interpolate the data points
        pos (torch.tensor): positions of the walkers Nbatch x 3*Nelec

    Returns:
        torch.tensor: interpolated values of the function evaluated at pos
    """

    nbatch, nelec, ndim = pos.shape[0], pos.shape[1]//3, 3
    return torch.tensor(interpfunc(pos.reshape(nbatch, nelec, ndim)))
