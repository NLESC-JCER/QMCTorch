import torch
from ..utils import fast_power

import torch
from ..utils import fast_power


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

    if not isinstance(derivative, list):
        derivative = [derivative]

    def _kernel():
        return rn * er

    def _first_derivative_kernel():
        if jacobian:
            nabla_rn = nabla_rn.sum(3)
            nabla_er = nabla_er.sum(3)
            return nabla_rn * er + rn * nabla_er
        else:
            return nabla_rn * \
                er.unsqueeze(-1) + rn.unsqueeze(-1) * nabla_er

    def _second_derivative_kernel():
        sum_xyz2 = (xyz*xyz).sum(3)

        R2 = R*R
        R3 = R2*R

        lap_rn = bas_n * (3 * R**(bas_n - 2) +
                          sum_xyz2 * (bas_n - 2) * R**(bas_n - 4))

        lap_er = bas_exp**2 * er * sum_xyz2 / R2 \
            - 2 * bas_exp * er * sum_xyz2 / R3

        return lap_rn * er + 2 * \
            (nabla_rn * nabla_er).sum(3) + rn * lap_er

    # computes the basic quantities
    rn = fast_power(R, bas_n)
    er = torch.exp(-bas_exp * R)

    # computes the grad
    if any(x in derivative for x in [1, 2]):
        nabla_rn = (bas_n * R**(bas_n - 2)).unsqueeze(-1) * xyz
        nabla_er = -(bas_exp * er).unsqueeze(-1) * \
            xyz / R.unsqueeze(-1)

    # prepare the output/kernel
    output = []
    fns = [_kernel,
           _first_derivative_kernel,
           _second_derivative_kernel]

    # compute the requested functions
    for d in derivative:
        output.append(fns[d]())

    if len(derivative) == 1:
        return output[0]
    else:
        return output


def radial_gaussian(R, bas_n, bas_exp, xyz=None, derivative=[0], jacobian=True):
    """Compute the radial part of GTOs (or its derivative).

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

    if not isinstance(derivative, list):
        derivative = [derivative]

    def _kernel():
        return rn * er

    def _first_derivative_kernel():
        if jacobian:
            nabla_rn_sum = nabla_rn.sum(3)
            nabla_er_sum = nabla_er.sum(3)
            return nabla_rn_sum * er + rn * nabla_er_sum
        else:
            return nabla_rn * \
                er.unsqueeze(-1) + rn.unsqueeze(-1) * nabla_er

    def _second_derivative_kernel():
        lap_rn = bas_n * (3 * R**(bas_n - 2)
                          + (xyz**2).sum(3) * (bas_n - 2) * R**(bas_n - 4))

        lap_er = 4 * bas_exp**2 * (xyz**2).sum(3) * er \
            - 6 * bas_exp * er

        return lap_rn * er + 2 * \
            (nabla_rn * nabla_er).sum(3) + rn * lap_er

    # computes the basic  quantities
    rn = fast_power(R, bas_n)
    er = torch.exp(-bas_exp * R*R)

    # computes the grads
    if any(x in derivative for x in [1, 2]):
        nabla_rn = (bas_n * R**(bas_n - 2)).unsqueeze(-1) * xyz
        nabla_er = -2 * (bas_exp * er).unsqueeze(-1) * xyz

    # prepare the output/function calls
    output = []
    fns = [_kernel,
           _first_derivative_kernel,
           _second_derivative_kernel]

    # compute the requested derivatives
    for d in derivative:
        output.append(fns[d]())

    if len(derivative) == 1:
        return output[0]
    else:
        return output
