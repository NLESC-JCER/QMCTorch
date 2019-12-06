import torch


def GradSphericalHarmonics(xyz, l, m):
    '''Compute the gradient of the Real Spherical Harmonics of the AO.
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, distance component of each
              point from each RBF center
        l : array(Nrbf) l quantum number
        m : array(Nrbf) m quantum number
    Returns:
        Y array (Nbatch,Nelec,Nrbf,3) : value of each grad SH at each point
    '''

    Y = torch.zeros(xyz.shape)

    # l=0
    ind = (l == 0).nonzero().view(-1)
    Y[:, :, ind, :] = _grad_spherical_harmonics_l0(xyz[:, :, ind, :])

    # l=1
    indl = (l == 1)
    if torch.any(indl):
        for mval in [-1, 0, 1]:
            indm = (m == mval)
            ind = (indl*indm).nonzero().view(-1)
            if len(ind > 0):
                # _tmp = _grad_spherical_harmonics_l1(xyz[:, :, ind, :], mval)
                Y[:, :, ind, :] = _grad_spherical_harmonics_l1(
                    xyz[:, :, ind, :], mval)

    # l=2
    indl = (l == 2)
    if torch.any(indl):
        for mval in [-2, -1, 0, 1, 2]:
            indm = (m == mval)
            ind = (indl*indm).nonzero().view(-1)
            if len(ind > 0):
                Y[:, :, ind, :] = _grad_spherical_harmonics_l2(
                    xyz[:, :, ind, :], mval)

    return Y


def _grad_spherical_harmonics_l0(xyz):
    ''' Compute the nabla of l=0 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
    Returns
        \nabla * Y00 = 0
    '''
    return torch.zeros(xyz.shape)


def _grad_spherical_harmonics_l1(xyz, m):
    ''' Compute the nabla of 1-1 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
        m : second quantum number (-1,0,1)
    Returns
        \nabla Y0-1 = \sqrt(3 / (4\pi)) ( 1/r^3 * [-yx, x^2+z^2, -yz]  ) (m=-1)
        \nabla Y00  = \sqrt(3 / (4\pi)) ( 1/r^3 * [-zx, -zy, x^2+y^2]  ) (m= 0)
        \nabla Y01  = \sqrt(3 / (4\pi)) ( 1/r^3 * [y^2+z^2, -xy, -xz]  ) (m=-1)
    '''

    r = torch.sqrt((xyz**2).sum(3))
    r3 = r**3
    c = 0.4886025119029199
    p = (c/r3).unsqueeze(-1)

    if m == -1:
        return p * (torch.stack([-xyz[:, :, :, 1]*xyz[:, :, :, 0],
                                 xyz[:, :, :, 0]**2+xyz[:, :, :, 2]**2,
                                 -xyz[:, :, :, 1]*xyz[:, :, :, 2]],
                                dim=-1))
    if m == 0:

        return p * (torch.stack([-xyz[:, :, :, 2]*xyz[:, :, :, 0],
                                 -xyz[:, :, :, 2]*xyz[:, :, :, 1],
                                 xyz[:, :, :, 0]**2+xyz[:, :, :, 1]**2],
                                dim=-1))
    if m == 1:
        return p * (torch.stack([xyz[:, :, :, 1]**2+xyz[:, :, :, 2]**2,
                                 -xyz[:, :, :, 0]*xyz[:, :, :, 1],
                                 -xyz[:, :, :, 0]*xyz[:, :, :, 2]],
                                dim=-1))


def _grad_spherical_harmonics_l2(xyz, m):
    """Compute the nabla of l=2 Spherical Harmonics
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
    r2 = r**2
    r3 = r**3
    r4 = r**4

    x = xyz[:, :, :, 0]
    y = xyz[:, :, :, 1]
    z = xyz[:, :, :, 2]

    if m == -2:
        c0 = 0.31539156525252005
        p = (c0/r4).unsqueeze(-1)
        return p * (torch.stack([y*(-x**2+y**2+z**2),
                                 x*(-y**2+x**2+z**2),
                                 -2*xyz.prod(-1)],
                                dim=-1))
    if m == -1:
        c0 = 0.31539156525252005
        p = (c0/r4).unsqueeze(-1)
        return p * (torch.stack([-2*xyz.prod(-1),
                                 z*(-y**2+x**2+z**2),
                                 y*(-z**2+x**2+y**2)],
                                dim=-1))
    if m == 0:
        c0 = 0.31539156525252005
        p = (c0/r4).unsqueeze(-1)
        return p * (torch.stack([-6*x*z*z,
                                 -6*y*z*z,
                                 6*x*x*z+6*y*y*z],
                                dim=-1))

    if m == 1:
        c0 = 0.31539156525252005
        p = (c0/r4).unsqueeze(-1)
        return p * (torch.stack([z*(-x*x+y*y+z*z),
                                 -2*xyz.prod(-1),
                                 x*(x*x+y*y-z*z)],
                                dim=-1))
    if m == 2:
        c2 = 0.5462742152960396
        p = (c0/r4).unsqueeze(-1)
        return p * (torch.stack([4*x*y*y + 2*x*z*z,
                                 -4*x*x*y - 2*y*z*z,
                                 -2*z*(x*x - y*y)],
                                dim=-1))


if __name__ == "__main__":

    Nbatch = 10
    Nelec = 5
    Nrbf = 3
    Ndim = 3

    xyz = torch.rand(Nbatch, Nelec, Nrbf, Ndim)

    l = torch.tensor([1, 1, 1])
    m = torch.tensor([0, 1, -1])

    Y = GradSphericalHarmonics(xyz, l, m)
