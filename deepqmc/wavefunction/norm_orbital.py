import torch

def norm_slater_spherial(bas_n, bas_exp):
    """ Normalization of STOs
    [1] www.theochem.ru.nl/~pwormer/Knowino/knowino.org/wiki/Slater_orbital.html
    [2] C Filippi, Multiconf wave functions for QMC of first row diatomic molecules, 
        JCP 105, 213 1996
    [3] Monte Carlo Methods in Ab Inition Quantum Chemistry, B.L. Hammond

    Returns:
        torch.tensor -- normalization factor
    """
    nfact = torch.tensor([np.math.factorial(2*n)
                            for n in bas_n], dtype=torch.get_default_dtype())
    return (2*bas_exp)**bas_n * torch.sqrt(2*bas_exp / nfact)

def norm_gaussian_spherical(bas_n, bas_exp):
    """ Normlization of GTOs.
    [1] Computational Quantum Chemistry: An interactive Intrduction to basis set theory 
        eq : 1.14 page 23.'''

    Returns:
        torch.tensor -- normalization factor
    """

    from scipy.special import factorial2 as f2

    bas_n = bas_n+1.
    exp1 = 0.25*(2.*bas_n+1.)

    A = bas_exp**exp1
    B = 2**(2.*bas_n+3./2)
    C = torch.tensor(f2(2*bas_n.int()-1)*np.pi **
                        0.5).type(torch.get_default_dtype())

    return torch.sqrt(B/C)*A


def norm_slater_cartesian(bas_n, bas_exp):
    raise NotImplementedError('TO DO')

def norm_gaussian_cartesian(bas_n, bas_exp):
    raise NotImplementedError('TO DO')