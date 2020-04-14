import torch
import numpy as np


def atomic_orbital_norm(basis):
    """Comptues the norm of a given function

    Arguments:
        basis {namespace} -- basis namespace
    """

    # spherical
    if basis.harmonics_type == 'sph':

        if basis.radial_type == 'sto':
            return norm_slater_spherial(basis.bas_n, basis.bas_exp)

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


def norm_slater_spherial(bas_n, bas_exp):
    """ Normalization of STOs
    [1] www.theochem.ru.nl/~pwormer/Knowino/knowino.org/wiki/Slater_orbital.html
    [2] C Filippi, Multiconf wave functions for QMC of first row diatomic molecules,
        JCP 105, 213 1996
    [3] Monte Carlo Methods in Ab Inition Quantum Chemistry, B.L. Hammond

    Returns:
        torch.tensor -- normalization factor
    """
    nfact = torch.tensor([np.math.factorial(2 * n)
                          for n in bas_n], dtype=torch.get_default_dtype())
    return (2 * bas_exp)**bas_n * torch.sqrt(2 * bas_exp / nfact)


def norm_gaussian_spherical(bas_n, bas_exp):
    """ Normlization of GTOs.
    [1] Computational Quantum Chemistry: An interactive Intrduction to basis set theory
        eq : 1.14 page 23.'''

    Returns:
        torch.tensor -- normalization factor
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
    """normaliation of cartesian slater
    Monte Carlo Methods in Ab Initio Quantum Chemistry page 279

    Arguments:
        a {[type]} -- exponent of x
        b {[type]} --  exponent of y
        c {[type]} --  exponent of z
        n {[type]} --  exponent of r
        exp {[type]} -- coefficient of the expo
    """
    from scipy.special import factorial2 as f2

    l = a + b + c + n + 1.
    lfact = torch.tensor([np.math.factorial(2 * i)
                          for i in l]).type(torch.get_default_dtype())
    prefact = 4 * np.pi * lfact / ((2 * exp)**(2 * l + 1))
    num = torch.tensor(f2(2 *
                          a.astype('int') -
                          1) *
                       f2(2 *
                          b.astype('int') -
                          1) *
                       f2(2 *
                          c.astype('int') -
                          1)).type(torch.get_default_dtype())
    denom = torch.tensor(
        f2((2 * a + 2 * b + 2 * c + 1).astype('int'))).type(torch.get_default_dtype())

    return torch.sqrt(1. / (prefact * num / denom))


def norm_gaussian_cartesian(a, b, c, exp):
    """normaliation of cartesian gaussian
    Monte Carlo Methods in Ab Initio Quantum Chemistry page 280

    Arguments:
        a {[type]} -- exponent of x
        b {[type]} --  exponent of y
        c {[type]} --  exponent of z
        exp {[type]} -- coefficient of the expo
    """

    from scipy.special import factorial2 as f2

    pref = torch.tensor((2 * exp / np.pi)**(0.75))
    x = (4 * exp)**(a / 2) / torch.sqrt(torch.tensor(f2((2 *
                                                         a - 1).astype('int')))).type(torch.get_default_dtype())
    y = (4 * exp)**(b / 2) / torch.sqrt(torch.tensor(f2((2 *
                                                         b - 1).astype('int')))).type(torch.get_default_dtype())
    z = (4 * exp)**(c / 2) / torch.sqrt(torch.tensor(f2((2 *
                                                         c - 1).astype('int')))).type(torch.get_default_dtype())
    return pref * x * y * z
