import torch
import numpy as np


def atomic_orbital_norm(basis):
    """Computes the norm of the atomic orbitals

    Args:
        basis (Namespace): basis object of the Molecule instance 

    Returns:
        torch.tensor: Norm of the atomic orbitals

    Examples::
        >>> mol = Molecule('h2.xyz', basis='dzp', calculator='adf')
        >>> norm = atomic_orbital_norm(mol.basis)
    """

    # spherical
    if basis.harmonics_type == 'sph':

        if basis.radial_type == 'sto':
            return norm_slater_spherical(basis.bas_n, basis.bas_exp)

        elif basis.radial_type == 'gto':
            return norm_gaussian_spherical(basis.bas_n, basis.bas_exp)

    # cartesian
    elif basis.harmonics_type == 'cart':

        if basis.radial_type == 'sto':
            return norm_slater_cartesian(
                basis.bas_kx,
                basis.bas_ky,
                basis.bas_kz,
                basis.bas_kr,
                basis.bas_exp)

        if basis.radial_type == 'gto':
            return norm_gaussian_cartesian(
                basis.bas_kx, basis.bas_ky, basis.bas_kz, basis.bas_exp)


def norm_slater_spherical(bas_n, bas_exp):
    """Normalization of STOs with Sphecrical Harmonics. \n
     * www.theochem.ru.nl/~pwormer/Knowino/knowino.org/wiki/Slater_orbital \n
     * C Filippi,  JCP 105, 213 1996 \n
     * Monte Carlo Methods in Ab Inition Quantum Chemistry, B.L. Hammond

    Args:
        bas_n (torch.tensor): prinicpal quantum number
        bas_exp (torch.tensor): slater exponents

    Returns:
        torch.tensor: normalization factor
    """
    nfact = torch.tensor([np.math.factorial(2 * n)
                          for n in bas_n], dtype=torch.get_default_dtype())
    return (2 * bas_exp)**bas_n * torch.sqrt(2 * bas_exp / nfact)


def norm_gaussian_spherical(bas_n, bas_exp):
    """Normlization of GTOs with spherical harmonics. \n
     * Computational Quantum Chemistry: An interactive Intrduction to basis set theory \n
        eq : 1.14 page 23.

    Args:
        bas_n (torch.tensor): prinicpal quantum number
        bas_exp (torch.tensor): slater exponents

    Returns:
        torch.tensor: normalization factor
    """

    from scipy.special import factorial2 as f2

    bas_n = bas_n + 1.
    exp1 = 0.25 * (2. * bas_n + 1.)

    A = bas_exp**exp1
    B = 2**(2. * bas_n + 3. / 2)
    C = torch.tensor(f2(2 * bas_n.int() - 1) * np.pi **
                     0.5).type(torch.get_default_dtype())

    return torch.sqrt(B / C) * A


def norm_slater_cartesian(a, b, c, n, exp):
    """Normaliation of STos with cartesian harmonics. \n
     * Monte Carlo Methods in Ab Initio Quantum Chemistry page 279

    Args:
        a (torch.tensor): exponent of x
        b (torch.tensor): exponent of y
        c (torch.tensor): exponent of z
        n (torch.tensor): exponent of r
        exp (torch.tensor): Sater exponent 

    Returns:
        torch.tensor: normalization factor
    """
    from scipy.special import factorial2 as f2

    lvals = a + b + c + n + 1.
    lfact = torch.tensor([np.math.factorial(2 * i)
                          for i in lvals]).type(torch.get_default_dtype())
    prefact = 4 * np.pi * lfact / ((2 * exp)**(2 * lvals + 1))
    num = torch.tensor(f2(2 * a.astype('int') - 1) *
                       f2(2 * b.astype('int') - 1) *
                       f2(2 * c.astype('int') - 1)
                       ).type(torch.get_default_dtype())
    denom = torch.tensor(
        f2((2 * a + 2 * b + 2 * c + 1).astype('int')
           )).type(torch.get_default_dtype())

    return torch.sqrt(1. / (prefact * num / denom))


def norm_gaussian_cartesian(a, b, c, exp):
    """Normaliation of GTOs with cartesian harmonics. \n
     * Monte Carlo Methods in Ab Initio Quantum Chemistry page 279

    Args:
        a (torch.tensor): exponent of x
        b (torch.tensor): exponent of y
        c (torch.tensor): exponent of z
        exp (torch.tensor): Sater exponent 

    Returns:
        torch.tensor: normalization factor
    """

    from scipy.special import factorial2 as f2

    pref = torch.tensor((2 * exp / np.pi)**(0.75))
    am1 = (2 * a - 1).astype('int')
    x = (4 * exp)**(a / 2) / torch.sqrt(torch.tensor(f2(am1)))

    bm1 = (2 * b - 1).astype('int')
    y = (4 * exp)**(b / 2) / torch.sqrt(torch.tensor(f2(bm1)))

    cm1 = (2 * c - 1).astype('int')
    z = (4 * exp)**(c / 2) / torch.sqrt(torch.tensor(f2(cm1)))

    return (pref * x * y * z).type(torch.get_default_dtype())
