import torch


def radial_slater(R, bas_n, bas_exp, xyz=None, derivative=0, jacobian=True):
    """Compute the radial part of STOs (or its derivative).

    Args:
        R (torch.tensor): distance between each electron and each atom
        bas_n (torch.tensor): principal quantum number
        bas_exp (torch.tensor): exponents of the exponential

    Keyword Arguments:
        xyz (torch.tensor): positions of the electrons
                            (needed for derivative) (default: {None})
        derivative (int): degree of the derivative (default: {0})
        jacobian (bool): return the jacobian, i.e the sum of the gradients
                           (default: {True})

    Returns:
        torch.tensor: values of each orbital radial part at each position
    """

    if derivative == 0:
        return R**bas_n * torch.exp(-bas_exp * R)

    elif derivative > 0:

        rn = R**(bas_n)
        nabla_rn = (bas_n * R**(bas_n - 2)).unsqueeze(-1) * xyz

        er = torch.exp(-bas_exp * R)
        nabla_er = -(bas_exp * er).unsqueeze(-1) * \
            xyz / R.unsqueeze(-1)

        if derivative == 1:

            if jacobian:
                nabla_rn = nabla_rn.sum(3)
                nabla_er = nabla_er.sum(3)
                return nabla_rn * er + rn * nabla_er
            else:
                return nabla_rn * \
                    er.unsqueeze(-1) + rn.unsqueeze(-1) * nabla_er

        elif derivative == 2:

            sum_xyz2 = (xyz**2).sum(3)

            lap_rn = bas_n * (3 * R**(bas_n - 2) +
                              sum_xyz2 * (bas_n - 2) * R**(bas_n - 4))

            lap_er = bas_exp**2 * er * sum_xyz2 / R**2 \
                - 2 * bas_exp * er * sum_xyz2 / R**3

            return lap_rn * er + 2 * \
                (nabla_rn * nabla_er).sum(3) + rn * lap_er


def radial_gaussian(R, bas_n, bas_exp, xyz=None, derivative=0, jacobian=True):
    """ompute the radial part of GTOs (or its derivative).

    Args:
        R (torch.tensor): distance between each electron and each atom
        bas_n (torch.tensor): principal quantum number
        bas_exp (torch.tensor): exponents of the exponential

    Keyword Arguments:
        xyz (torch.tensor): positions of the electrons
                            (needed for derivative) (default: {None})
        derivative (int): degree of the derivative (default: {0})
        jacobian (bool): return the jacobian, i.e the sum of the gradients
                           (default: {True})

    Returns:
        torch.tensor: values of each orbital radial part at each position
    """
    if derivative == 0:
        return R**bas_n * torch.exp(-bas_exp * R**2)

    elif derivative > 0:

        rn = R**(bas_n)
        nabla_rn = (bas_n * R**(bas_n - 2)).unsqueeze(-1) * xyz

        er = torch.exp(-bas_exp * R**2)
        nabla_er = -2 * (bas_exp * er).unsqueeze(-1) * xyz

        if derivative == 1:
            if jacobian:
                nabla_rn = nabla_rn.sum(3)
                nabla_er = nabla_er.sum(3)
                return nabla_rn * er + rn * nabla_er
            else:
                return nabla_rn * \
                    er.unsqueeze(-1) + rn.unsqueeze(-1) * nabla_er

        elif derivative == 2:

            lap_rn = bas_n * (3 * R**(bas_n - 2)
                              + (xyz**2).sum(3) * (bas_n - 2) * R**(bas_n - 4))

            lap_er = 4 * bas_exp**2 * (xyz**2).sum(3) * er \
                - 6 * bas_exp * er

            return lap_rn * er + 2 * \
                (nabla_rn * nabla_er).sum(3) + rn * lap_er
