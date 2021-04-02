from time import time

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator


class InterpolateMolecularOrbitals:

    def __init__(self, wf):
        """Interpolation of the AO using a log grid centered on each atom."""
        self.wf = wf

    def __call__(self, pos, method='irreg', orb='occupied', **kwargs):

        if method == 'irreg':
            n = kwargs['n'] if 'n' in kwargs else 6
            return self.interpolate_mo_irreg_grid(pos, n=n, orb=orb)

        elif method == 'reg':
            rstr, bstr = 'resolution', 'border_length'
            res = kwargs[rstr] if rstr in kwargs else 0.1
            blength = kwargs[bstr] if bstr in kwargs else 2.
            return self.interpolate_mo_reg_grid(pos, res, blength, orb)

    def get_mo_max_index(self, orb):
        """Get the index of the highest MO to inlcude in the interpoaltion

        Args:
            orb (str): occupied or all

        Raises:
            ValueError: if orb not valid
        """

        if orb == 'occupied':
            self.mo_max_index = torch.stack(
                self.wf.configs).max().item()+1
        elif orb == 'all':
            self.mo_max_index = self.wf.mol.basis.nmo+1
        else:
            raise ValueError(
                'orb must occupied or all')

    def interpolate_mo_irreg_grid(self, pos, n, orb):
        """Interpolate the mo occupied in the configs.

        Args:
            pos (torch.tensor): sampling points (Nbatch, 3*Nelec)
            n (int, optional): Interpolation order. Defaults to 6.

        Returns:
            torch.tensor: mo values Nbatch, Nelec, Nmo
        """
        self.get_mo_max_index(orb)

        if not hasattr(self, 'interp_mo_func'):
            grid_pts = get_log_grid(self.wf.mol.atom_coords, n=n)

            def func(x):
                x = torch.as_tensor(x).type(torch.get_default_dtype())
                ao = self.wf.ao(x, one_elec=True)
                mo = self.wf.mo(self.wf.mo_scf(ao)).squeeze(1)
                return mo[:, :self.mo_max_index].detach()

            self.interp_mo_func = interpolator_irreg_grid(
                func, grid_pts)

        nbatch = pos.shape[0]
        mos = torch.zeros(nbatch, self.wf.mol.nelec,
                          self.wf.mol.basis.nmo)
        mos[:, :, :self.mo_max_index] = interpolate_irreg_grid(
            self.interp_mo_func, pos)
        return mos

    def interpolate_mo_reg_grid(self, pos,  res, blength, orb):
        """Interpolate the mo occupied in the configs.

        Args:
            pos (torch.tensor): sampling points (Nbatch, 3*Nelec)

        Returns:
            torch.tensor: mo values Nbatch, Nelec, Nmo
        """

        self.get_mo_max_index(orb)

        if not hasattr(self, 'interp_mo_func'):
            x, y, z = get_reg_grid(
                self.wf.mol.atom_coords, resolution=res, border_length=blength)

            def func(x):
                x = torch.as_tensor(x).type(torch.get_default_dtype())
                ao = self.wf.ao(x, one_elec=True)
                mo = self.wf.mo(self.wf.mo_scf(ao)).squeeze(1)
                return mo[:, :self.mo_max_index]

            self.interp_mo_func = interpolator_reg_grid(
                func, x, y, z)

        nbatch = pos.shape[0]
        mos = torch.zeros(nbatch, self.wf.mol.nelec,
                          self.wf.mol.basis.nmo)
        mos[:, :, :self.mo_max_index] = interpolate_reg_grid(
            self.interp_mo_func, pos)
        return mos


class InterpolateAtomicOrbitals:

    def __init__(self, wf):
        """Interpolation of the AO using a log grid centered on each atom."""
        self.wf = wf

    def __call__(self, pos, n=6, length=2):
        """Interpolate the AO.

        Args:
            pos (torch.tensor): positions of the walkers
            n (int, optional): number of points on each log axis. Defaults to 6.
            length (int, optional): half length of the grid. Defaults to 2.

        Returns:
            torch.tensor: Interpolated values
        """

        if not hasattr(self, 'interp_func'):
            t0 = time()
            self.get_interpolator()
            print('___', time()-t0)

        t0 = time()
        bas_coords = self.wf.ao.atom_coords.repeat_interleave(
            self.wf.ao.nao_per_atom, dim=0)  # <- we need the number of AO per atom not the number of BAS per atom
        print('___bas ', time()-t0)

        t0 = time()
        xyz = (pos.view(-1, self.wf.ao.nelec, 1, self.wf.ao.ndim) -
               bas_coords[None, ...]).detach().numpy()
        print('___ xyz', time()-t0)

        t0 = time()
        data = np.array([self.interp_func[iorb](xyz[:, :, iorb, :])
                         for iorb in range(self.wf.ao.norb)])
        print('___ data', time()-t0)

        return torch.as_tensor(data.transpose(1, 2, 0))

    def get_interpolator(self, n=6, length=2):
        """evaluate the interpolation function.

        Args:
            n (int, optional): number of points on each log axis. Defaults to 6.
            length (int, optional): half length of the grid. Defaults to 2.
        """

        xpts = logspace(n, length)
        nxpts = len(xpts)

        grid = np.stack(np.meshgrid(
            xpts, xpts, xpts, indexing='ij')).T.reshape(-1, 3)[:, [2, 1, 0]]

        def func(x):
            x = torch.as_tensor(x).type(torch.get_default_dtype())
            nbatch = x.shape[0]
            xyz = x.view(-1, 1, 1, 3).expand(-1,
                                             1, self.wf.ao.nbas, 3)
            r = torch.sqrt((xyz**2).sum(3))
            R = self.wf.ao.radial(r, self.wf.ao.bas_n,
                                  self.wf.ao.bas_exp)
            Y = self.wf.ao.harmonics(xyz)
            bas = R * Y
            bas = self.wf.ao.norm_cst * self.wf.ao.bas_coeffs * bas
            ao = torch.zeros(nbatch, self.wf.ao.nelec,
                             self.wf.ao.norb, device=self.wf.ao.device)
            ao.index_add_(2, self.wf.ao.index_ctr, bas)
            return ao

        data = func(grid).detach().numpy()
        data = data.reshape(nxpts, nxpts, nxpts, -1)

        self.interp_func = [RegularGridInterpolator((xpts, xpts, xpts),
                                                    data[..., i],
                                                    method='linear',
                                                    bounds_error=False,
                                                    fill_value=0.) for i in range(self.wf.ao.norb)]


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

    return torch.as_tensor(data)


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
    return torch.as_tensor(interpfunc(pos.reshape(nbatch, nelec, ndim)))
