import torch
from ...utils import fast_power


class Harmonics:

    def __init__(self, type, **kwargs):
        """Compute spherical or cartesian harmonics and their derivatives

        Args:
            type (str): harmonics type (cart or sph)

        Keyword Arguments:
            bas_l (torch.tensor): second quantum numbers (sph)
            bas_m (torch.tensor): third quantum numbers (sph)
            bas_kx (torch.tensor): x exponent (cart)
            bas_ky (torch.tensor): xy exponent (cart)
            bas_kz (torch.tensor): z exponent (cart)
            cuda (bool): use cuda (defaults False)

        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> harm = Harmonics(cart)
            >>> pos = torch.rand(100,6)
            >>> hvals = harm(pos)
            >>> dhvals = harm(pos,derivative=1)
        """

        self.type = type

        # check if we need cuda
        if 'cuda' not in kwargs:
            cuda = False
        else:
            cuda = kwargs['cuda']

        # select the device
        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # register parameters
        if self.type == 'sph':
            self.bas_l = torch.as_tensor(
                kwargs['bas_l']).to(self.device)
            self.bas_m = torch.as_tensor(
                kwargs['bas_m']).to(self.device)

        elif self.type == 'cart':

            self.bas_kx = torch.as_tensor(
                kwargs['bas_kx']).to(self.device)
            self.bas_ky = torch.as_tensor(
                kwargs['bas_ky']).to(self.device)
            self.bas_kz = torch.as_tensor(
                kwargs['bas_kz']).to(self.device)

            self.bas_k = torch.stack(
                (self.bas_kx, self.bas_ky, self.bas_kz)).transpose(0, 1)
            self.mask_bas_k0 = self.bas_k == 0
            self.mask_bas_k2 = self.bas_k == 2

    def __call__(self, xyz, derivative=[0], sum_grad=True, sum_hess=True):
        """Computes the cartesian or spherical harmonics

        Arguments:
            xyz {torch.tensor} -- coordinate of each electrons from each BAS
                                  center (Nbatch, Nelec, Nbas, Ndim)

        Keyword Arguments:
            derivative {int} -- order of the derivative (default: {0})
            sum_grad {bool} -- return the sum of th derivative if true and
                               grad if False (default: {True})
            sum_hess {bool} -- return the sum of the 2nd derivative if true and
                               grad if False (default: {True})

        Raises:
            ValueError: of type is unrecognized

        Returns:
            torch.tensor -- Values or gradient of the spherical harmonics
        """

        if self.type == 'cart':
            return CartesianHarmonics(xyz, self.bas_k, self.mask_bas_k0, self.mask_bas_k2,
                                      derivative, sum_grad, sum_hess)
        elif self.type == 'sph':
            return SphericalHarmonics(
                xyz, self.bas_l, self.bas_m, derivative, sum_grad, sum_hess)
        else:
            raise ValueError('Harmonics type should be cart or sph')


def CartesianHarmonics(xyz, k, mask0, mask2, derivative=[0],
                       sum_grad=True, sum_hess=True):
    r"""Computes Real Cartesian Harmonics

    .. math::
        Y = x^{k_x} \\times y^{k_y} \\times z^{k_z}

    Args:
        xyz (torch.tensor): distance between sampling points and orbital centers \n
                            size : (Nbatch, Nelec, Nbas, Ndim)
        k (torch.tensor): (kx,ky,kz) exponents
        mask0 (torch.tensor): precomputed mask of k=0
        mask2 (torch.tensor): precomputed mask of k=2
        derivative (int, optional): degree of the derivative. Defaults to 0.
        sum_grad (bool, optional): returns the sum of the derivative if True. Defaults to True.
        sum_hess (bool, optional): returns the sum of the 2nd derivative if True. Defaults to True.
    Returns:
        torch.tensor: values of the harmonics at the sampling points
    """

    if not isinstance(derivative, list):
        derivative = [derivative]

    def _kernel():
        return xyz_k.prod(-1)

    def _first_derivative_kernel():
        km1 = k-1
        km1[km1 < 0] = 0

        xyz_km1 = fast_power(xyz, km1)

        kx, ky, kz = k.transpose(0, 1)
        dx = kx * xyz_km1[..., 0] * xyz_k[..., 1] * xyz_k[..., 2]
        dy = ky * xyz_k[..., 0] * xyz_km1[..., 1] * xyz_k[..., 2]
        dz = kz * xyz_k[..., 0] * xyz_k[..., 1] * xyz_km1[..., 2]

        if sum_grad:
            return dx + dy + dz
        else:
            return torch.stack((dx, dy, dz), dim=-1)

    def _second_derivative_kernel():
        # prepare the exponets
        km2 = k - 2
        km2[km2 < 0] = 0

        xyz_km2 = fast_power(xyz, km2)

        kx, ky, kz = k.transpose(0, 1)

        d2x = kx*(kx-1) * xyz_km2[..., 0] * \
            xyz_k[..., 1] * xyz_k[..., 2]
        d2y = ky*(ky-1) * xyz_k[..., 0] * \
            xyz_km2[..., 1] * xyz_k[..., 2]
        d2z = kz*(kz-1) * xyz_k[..., 0] * \
            xyz_k[..., 1] * xyz_km2[..., 2]

        if sum_hess:
            return d2x + d2y + d2z
        else:
            return torch.stack((d2x, d2y, d2z), dim=-1)

    def _mixed_second_derivative_kernel():
        km1 = k-1
        km1[km1 < 0] = 0

        xyz_km1 = fast_power(xyz, km1)

        kx, ky, kz = k.transpose(0, 1)

        dxdy = kx * xyz_km1[..., 0] * ky * \
            xyz_km1[..., 1] * xyz_k[..., 2]
        dxdz = kx * xyz_km1[..., 0] * \
            xyz_k[..., 1] * kz * xyz_km1[..., 2]
        dydz = xyz_k[..., 0] * ky * \
            xyz_km1[..., 1] * kz * xyz_km1[..., 2]

        return torch.stack((dxdy, dxdz, dydz), dim=-1)

    # computes the power of the xyz
    xyz_k = fast_power(xyz, k,  mask0, mask2)

    # compute the outputs
    fns = [_kernel,
           _first_derivative_kernel,
           _second_derivative_kernel,
           _mixed_second_derivative_kernel]

    output = []
    for d in derivative:
        output.append(fns[d]())

    if len(derivative) == 1:
        return output[0]
    else:
        return output


def SphericalHarmonics(xyz, l, m, derivative=0, sum_grad=True, sum_hess=True):
    r"""Compute the Real Spherical Harmonics of the AO.

    Args:
        xyz (torch.tensor): distance between sampling points and orbital centers \n
                            size : (Nbatch, Nelec, Nbas, Ndim)
        l (torch.tensor):  l quantum number
        m (torch.tensor):  m quantum number

    Returns:
        Y (torch.tensor): value of each harmonics at each points (or derivative) \n
                          size : (Nbatch,Nelec,Nrbf) for sum_grad=True \n
                          size : (Nbatch,Nelec,Nrbf, Ndim) for sum_grad=False
    """
    if not sum_hess:
        raise NotImplementedError(
            'SphericalHarmonics cannot return individual component of the laplacian')
    if derivative > 2:
        raise NotImplementedError(
            "Spherical Harmonics only accpet derivative=0,1,2 (%d found)" % derivative)

    if sum_grad:
        return get_spherical_harmonics(xyz, l, m, derivative)
    else:
        if derivative != 1:
            raise ValueError(
                'Gradient of the spherical harmonics require derivative=1')
        return get_grad_spherical_harmonics(xyz, l, m)


def get_spherical_harmonics(xyz, lval, m, derivative):
    r"""Compute the Real Spherical Harmonics of the AO.

    Args:
        xyz (torch.tensor): distance between sampling points and orbital centers \n
                            size : (Nbatch, Nelec, Nbas, Ndim)
        l (torch.tensor): l quantum number
        m (torch.tensor): m quantum number

    Returns:
        Y (torch.tensor): value of each harmonics at each points (or derivative) \n
                          size : (Nbatch,Nelec,Nrbf)
    """

    Y = torch.zeros_like(xyz[..., 0])

    # l=0
    ind = (lval == 0).nonzero().view(-1)
    if derivative == 0:
        Y[:, :, ind] = _spherical_harmonics_l0(xyz[:, :, ind, :])
    if derivative == 1:
        Y[:, :, ind] = _nabla_spherical_harmonics_l0(
            xyz[:, :, ind, :])

    # l=1
    indl = (lval == 1)
    if torch.any(indl):
        for mval in [-1, 0, 1]:
            indm = (m == mval)
            ind = (indl * indm).nonzero().view(-1)
            if len(ind > 0):
                if derivative == 0:
                    Y[:, :, ind] = _spherical_harmonics_l1(
                        xyz[:, :, ind, :], mval)
                if derivative == 1:
                    Y[:, :, ind] = _nabla_spherical_harmonics_l1(
                        xyz[:, :, ind, :], mval)
                if derivative == 2:
                    Y[:, :, ind] = _lap_spherical_harmonics_l1(
                        xyz[:, :, ind, :], mval)

    # l=2
    indl = (lval == 2)
    if torch.any(indl):
        for mval in [-2, -1, 0, 1, 2]:
            indm = (m == mval)
            ind = (indl * indm).nonzero().view(-1)
            if len(ind > 0):
                if derivative == 0:
                    Y[:, :, ind] = _spherical_harmonics_l2(
                        xyz[:, :, ind, :], mval)
                if derivative == 1:
                    Y[:, :, ind] = _nabla_spherical_harmonics_l2(
                        xyz[:, :, ind, :], mval)
                if derivative == 2:
                    Y[:, :, ind] = _lap_spherical_harmonics_l2(
                        xyz[:, :, ind, :], mval)

    return Y


def get_grad_spherical_harmonics(xyz, lval, m):
    r"""Compute the gradient of the Real Spherical Harmonics of the AO.

    Args:
        xyz (torch.tensor): distance between sampling points and orbital centers \n
                            size : (Nbatch, Nelec, Nbas, Ndim)
        l (torch.tensor): l quantum number
        m (torch.tensor): m quantum number

    Returns:
        Y (torch.tensor): value of each harmonics at each points (or derivative) \n
                          size : (Nbatch,Nelec,Nrbf,3)
    """

    Y = torch.zeros_like(xyz)

    # l=0
    ind = (lval == 0).nonzero().view(-1)
    Y[:, :, ind, :] = _grad_spherical_harmonics_l0(xyz[:, :, ind, :])

    # l=1
    indl = (lval == 1)
    if torch.any(indl):
        for mval in [-1, 0, 1]:
            indm = (m == mval)
            ind = (indl * indm).nonzero().view(-1)
            if len(ind > 0):
                # _tmp = _grad_spherical_harmonics_l1(xyz[:, :, ind, :], mval)
                Y[:, :, ind, :] = _grad_spherical_harmonics_l1(
                    xyz[:, :, ind, :], mval)

    # l=2
    indl = (lval == 2)
    if torch.any(indl):
        for mval in [-2, -1, 0, 1, 2]:
            indm = (m == mval)
            ind = (indl * indm).nonzero().view(-1)
            if len(ind > 0):
                Y[:, :, ind, :] = _grad_spherical_harmonics_l2(
                    xyz[:, :, ind, :], mval)

    return Y

# =============== L0


def _spherical_harmonics_l0(xyz):
    r"""Compute the l=0 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
    Returns
        Y00 = 1/2 \sqrt(1 / \pi)
    """

    return 0.2820948 * torch.ones_like(xyz[..., 0])


def _nabla_spherical_harmonics_l0(xyz):
    r"""Compute the nabla of l=0 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
    Returns
        \nabla * Y00 = 0
    """
    return torch.zeros_like(xyz[..., 0])


def _grad_spherical_harmonics_l0(xyz):
    r"""Compute the nabla of l=0 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
    Returns
        \nabla * Y00 = 0
    """
    return torch.zeros_like(xyz)


def _lap_spherical_harmonics_l0(xyz):
    r"""Compute the laplacian of l=0 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
    Returns
        \nabla^2 * Y00 = 0
    """
    return torch.zeros_like(xyz[..., 0])

# =============== L1


def _spherical_harmonics_l1(xyz, m):
    r"""Compute the 1-1 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
        m : second quantum number (-1,0,1)
    Returns
        Y0-1 = \sqrt(3 / (4\pi)) y/r. (m=-1)
        Y00  = \sqrt(3 / (4\pi)) z/r (m=0)
        Y01  = \sqrt(3 / (4\pi)) x/r. (m=1)
    """
    index = {-1: 1, 0: 2, 1: 0}
    r = torch.sqrt((xyz**2).sum(3))
    c = 0.4886025119029199
    return c * xyz[:, :, :, index[m]] / r


def _nabla_spherical_harmonics_l1(xyz, m):
    r"""Compute the nabla of 1-1 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
        m : second quantum number (-1,0,1)
    Returns
        \nabla Y0-1 = \sqrt(3 / (4\pi)) ( 1/r - y (x+y+z)/r^3 ) (m=-1)
        \nabla Y00  = \sqrt(3 / (4\pi)) ( 1/r - z (x+y+z)/r^3 ) (m= 0)
        \nabla Y01  = \sqrt(3 / (4\pi)) ( 1/r - x (x+y+z)/r^3 ) (m= 1)
    """
    index = {-1: 1, 0: 2, 1: 0}
    r = torch.sqrt((xyz**2).sum(3))
    r3 = r**3
    c = 0.4886025119029199
    return c * (1. / r - xyz[:, :, :, index[m]] * xyz.sum(3) / r3)


def _grad_spherical_harmonics_l1(xyz, m):
    r"""Compute the nabla of 1-1 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
        m : second quantum number (-1,0,1)
    Returns
        \nabla Y0-1 = \sqrt(3 / (4\pi)) ( 1/r^3 * [-yx, x^2+z^2, -yz]  ) (m=-1)
        \nabla Y00  = \sqrt(3 / (4\pi)) ( 1/r^3 * [-zx, -zy, x^2+y^2]  ) (m= 0)
        \nabla Y01  = \sqrt(3 / (4\pi)) ( 1/r^3 * [y^2+z^2, -xy, -xz]  ) (m=-1)
    """

    r = torch.sqrt((xyz**2).sum(3))
    r3 = r**3
    c = 0.4886025119029199
    p = (c / r3).unsqueeze(-1)

    if m == -1:
        return p * (torch.stack([-xyz[:, :, :, 1] * xyz[:, :, :, 0],
                                 xyz[:, :, :, 0]**2 +
                                 xyz[:, :, :, 2]**2,
                                 -xyz[:, :, :, 1] * xyz[:, :, :, 2]],
                                dim=-1))
    if m == 0:

        return p * (torch.stack([-xyz[:, :, :, 2] * xyz[:, :, :, 0],
                                 -xyz[:, :, :, 2] * xyz[:, :, :, 1],
                                 xyz[:, :, :, 0]**2 + xyz[:, :, :, 1]**2],
                                dim=-1))
    if m == 1:
        return p * (torch.stack([xyz[:, :, :, 1]**2 + xyz[:, :, :, 2]**2,
                                 -xyz[:, :, :, 0] * xyz[:, :, :, 1],
                                 -xyz[:, :, :, 0] * xyz[:, :, :, 2]],
                                dim=-1))


def _lap_spherical_harmonics_l1(xyz, m):
    r"""Compute the laplacian of 1-1 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
        m : second quantum number (-1,0,1)
    Returns
        Y0-1 = \sqrt(3 / (4\pi)) ( -2y/r^3 ) (m=-1)
        Y00  = \sqrt(3 / (4\pi)) ( -2z/r^3 ) (m= 0)
        Y01  = \sqrt(3 / (4\pi)) ( -2x/r^3 ) (m= 1)
    """
    index = {-1: 1, 0: 2, 1: 0}
    r = torch.sqrt((xyz**2).sum(3))
    r3 = r**3
    c = 0.4886025119029199
    return c * (- 2 * xyz[:, :, :, index[m]] / r3)

# =============== L2


def _spherical_harmonics_l2(xyz, m):
    r"""Compute the l=2 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
        m : second quantum number (-2,-1,0,1,2)
    Returns
        Y2-2 = 1/2\sqrt(15/\pi) xy/r^2
        Y2-1 = 1/2\sqrt(15/\pi) yz/r^2
        Y20  = 1/4\sqrt(5/\pi) (-x^2-y^2+2z^2)/r^2
        Y21  = 1/2\sqrt(15/\pi) zx/r^2
        Y22  = 1/4\sqrt(15/\pi) (x*x-y*y)/r^2
    """

    r2 = (xyz**2).sum(-1)

    if m == 0:
        c0 = 0.31539156525252005
        return c0 * (-xyz[:, :, :, 0]**2 - xyz[:, :, :, 1]
                     ** 2 + 2 * xyz[:, :, :, 2]**2) / r2
    if m == 2:
        c2 = 0.5462742152960396
        return c2 * (xyz[:, :, :, 0]**2 - xyz[:, :, :, 1]**2) / r2
    else:
        cm = 1.0925484305920792
        index = {-2: [0, 1], -1: [1, 2], 1: [2, 0]}
        return cm * xyz[:, :, :, index[m][0]] * \
            xyz[:, :, :, index[m][1]] / r2


def _nabla_spherical_harmonics_l2(xyz, m):
    r"""Compute the nabla of l=2 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
        m : second quantum number (-2,-1,0,1,2)
    Returns
        Y2-2 = 1/2\sqrt(15/\pi) (x+y)/r^2 - 2 * xy (x+y+z)/r^3
        Y2-1 = 1/2\sqrt(15/\pi) (y+z)/r^2 - 2 * yz (x+y+z)/r^3
        Y20  = 1/4\sqrt(5/\pi) ( (-2x-2y+4z)/r^2 - \
               2 *(-xx - yy + 2zz) * (x+y+z)/r3 )
        Y21  = 1/2\sqrt(15/\pi) (x+z)/r^2 - 2 * xz (x+y+z)/r^3
        Y22  = 1/4\sqrt(15/\pi)  ( 2(x-y)/r^2 - 2 *(xx-yy)(x+y+z)/r^3  )
    """

    r = torch.sqrt((xyz**2).sum(3))
    r2 = r**2
    r3 = r**3

    if m == 0:
        c0 = 0.31539156525252005
        return c0 * ((- 2 * xyz[:, :, :, 0] - 2 * xyz[:, :, :, 1] + 4 * xyz[:, :, :, 2]) / r2
                     - 2 * (-xyz[:, :, :, 0]**2 - xyz[:, :, :, 1]**2 + 2 * xyz[:, :, :, 2]**2) * xyz.sum(3) / r3)
    if m == 2:
        c2 = 0.5462742152960396
        return c2 * (2 * (xyz[:, :, :, 0] - xyz[:, :, :, 1]) / r2 - 2 * (xyz[:, :, :, 0]**2
                                                                         - xyz[:, :, :, 1]**2) * xyz.sum(3) / r3)
    else:
        cm = 1.0925484305920792
        index = {-2: [0, 1], -1: [1, 2], 1: [2, 0]}
        return cm * ((xyz[:, :, :, index[m][0]] + xyz[:, :, :, index[m][1]]) / r2
                     - 2 * xyz[:, :, :, index[m][0]] * xyz[:, :, :, index[m][1]] * xyz.sum(3) / r3)


def _grad_spherical_harmonics_l2(xyz, m):
    r"""Compute the nabla of l=2 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
        m : second quantum number (-2,-1,0,1,2)

    Returns
        Y2-2 = 1/2 \sqrt(15/\pi) 1./r4 ([y(-xx+yy+zz, x(-yy+xx+zz,-2xyz))])
        Y2-1 = 1/2 \sqrt(15/\pi)
        Y20  = 1/4 \sqrt(5/\pi)
        Y21  = 1/2 \sqrt(15/\pi)
        Y22  = 1/4 \sqrt(15/\pi)
    """

    r = torch.sqrt((xyz**2).sum(3))
    r4 = r**4

    x = xyz[:, :, :, 0]
    y = xyz[:, :, :, 1]
    z = xyz[:, :, :, 2]

    if m == -2:
        c0 = 0.31539156525252005
        p = (c0 / r4).unsqueeze(-1)
        return p * (torch.stack([y * (-x**2 + y**2 + z**2),
                                 x * (-y**2 + x**2 + z**2),
                                 -2 * xyz.prod(-1)],
                                dim=-1))
    if m == -1:
        c0 = 0.31539156525252005
        p = (c0 / r4).unsqueeze(-1)
        return p * (torch.stack([-2 * xyz.prod(-1),
                                 z * (-y**2 + x**2 + z**2),
                                 y * (-z**2 + x**2 + y**2)],
                                dim=-1))
    if m == 0:
        c0 = 0.31539156525252005
        p = (c0 / r4).unsqueeze(-1)
        return p * (torch.stack([-6 * x * z * z,
                                 -6 * y * z * z,
                                 6 * x * x * z + 6 * y * y * z],
                                dim=-1))

    if m == 1:
        c0 = 0.31539156525252005
        p = (c0 / r4).unsqueeze(-1)
        return p * (torch.stack([z * (-x * x + y * y + z * z),
                                 -2 * xyz.prod(-1),
                                 x * (x * x + y * y - z * z)],
                                dim=-1))
    if m == 2:
        c0 = 0.5462742152960396
        p = (c0 / r4).unsqueeze(-1)
        return p * (torch.stack([4 * x * y * y + 2 * x * z * z,
                                 -4 * x * x * y - 2 * y * z * z,
                                 -2 * z * (x * x - y * y)],
                                dim=-1))


def _lap_spherical_harmonics_l2(xyz, m):
    r"""Compute the nabla of l=2 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
        m : second quantum number (-2,-1,0,1,2)
    Returns
        Y2-2 = 1/2\sqrt(15/\pi) -6xy/r^4
        Y2-1 = 1/2\sqrt(15/\pi) -6yz/r^4

        Y20  = 1/4\sqrt(5/\pi) ( 6/r6 * (xx+yy)^2 - zz * (xx + yy -2zz))

        Y21  = 1/2\sqrt(15/\pi) -6zx/r^4
        Y22  = 1/4\sqrt(15/\pi)  ( 6/r6 * ( zz*(yy-xx)  +y^4 - x^4  )  )
    """

    r = torch.sqrt((xyz**2).sum(3))
    r4 = r**4
    r6 = r**6

    if m == 0:
        c0 = 0.31539156525252005
        xyz2 = xyz**2
        return c0 * (6 / r6 * (xyz2[:, :, :, :2].sum(-1))**2 - xyz2[:, :, :, 2] * (xyz2[:, :, :, 0]
                                                                                   + xyz2[:, :, :, 1] - 2 * xyz2[:, :, :, 2]))
    if m == 2:
        c2 = 0.5462742152960396
        xyz2 = xyz**2
        return c2 * (6 / r6 * xyz2[:, :, :, 2] * (xyz2[:, :, :, 1] - xyz2[:, :, :, 0])
                     + xyz2[:, :, :, 1]**2 - xyz2[:, :, :, 0]**2)
    else:
        cm = 1.0925484305920792
        index = {-2: [0, 1], -1: [1, 2], 1: [2, 0]}
        return cm * (- 6 * xyz[:, :, :, index[m][0]]
                     * xyz[:, :, :, index[m][1]] / r4)
