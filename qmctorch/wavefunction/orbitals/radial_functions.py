import torch
from typing import Union, List, Callable
from ...utils import fast_power


def radial_slater(
    R: torch.Tensor,  # distance between each electron and each atom
    bas_n: torch.Tensor,  # principal quantum number
    bas_exp: torch.Tensor,  # exponents of the exponential
    xyz: torch.Tensor = None,  # positions of the electrons
    derivative: int = 0,  # degree of the derivative
    sum_grad: bool = True,  # return the sum_grad, i.e the sum of the gradients
    sum_hess: bool = True,  # return the sum_hess, i.e the sum of the diag hessian
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute the radial part of STOs (or its derivative).

    .. math:
        sto = r^n exp(-\alpha |r|)

    Args:
        R (torch.tensor): distance between each electron and each atom
        bas_n (torch.tensor): principal quantum number
        bas_exp (torch.tensor): exponents of the exponential

    Keyword Arguments:
        xyz (torch.tensor): positions of the electrons
                            (needed for derivative) (default: {None})
        derivative (int): degree of the derivative (default: {0})
                          0 : value of the function
                          1 : first derivative
                          2 : pure second derivative
                          3 : mixed second derivative
        sum_grad (bool): return the sum_grad, i.e the sum of the gradients
                           (default: {True})
        sum_hess (bool): return the sum_hess, i.e the sum of the diag hessian
                           (default: {False})

    Returns:
        torch.tensor: values of each orbital radial part at each position
    """

    if not isinstance(derivative, list):
        derivative = [derivative]

    def _kernel() -> torch.Tensor:
        """Return the kernel."""
        return rn * er

    def _first_derivative_kernel() -> torch.Tensor:
        """Return the first derivative."""
        if sum_grad:
            nabla_rn_sum = nabla_rn.sum(3)
            nabla_er_sum = nabla_er.sum(3)
            return nabla_rn_sum * er + rn * nabla_er_sum
        else:
            return nabla_rn * er.unsqueeze(-1) + rn.unsqueeze(-1) * nabla_er

    def _second_derivative_kernel() -> torch.Tensor:
        """Return the pure second derivative i.e. d^2/dx^2"""
        if sum_hess:
            lap_rn = nRnm2 * (bas_n + 1)
            lap_er = bexp_er * (bas_exp - 2.0 / R)

            return lap_rn * er + 2 * (nabla_rn * nabla_er).sum(3) + rn * lap_er
        else:
            xyz2 = xyz * xyz
            xyz2 = xyz2 / xyz2.sum(-1, keepdim=True)

            lap_rn = nRnm2.unsqueeze(-1) * (1.0 + (bas_n - 2).unsqueeze(-1) * xyz2)

            lap_er = bexp_er.unsqueeze(-1) * (
                bas_exp.unsqueeze(-1) * xyz2 + (-1 + xyz2) / R.unsqueeze(-1)
            )

            return (
                lap_rn * er.unsqueeze(-1)
                + 2 * (nabla_rn * nabla_er)
                + rn.unsqueeze(-1) * lap_er
            )

    def _mixed_second_derivative_kernel() -> torch.Tensor:
        """Returns the mixed second derivative i.e. d^2/dxdy.
        where x and y are coordinate of the same electron."""

        mix_prod = xyz[..., [[0, 1], [0, 2], [1, 2]]].prod(-1)
        nRnm4 = nRnm2 / (xyz * xyz).sum(-1)

        lap_rn = ((bas_n - 2) * nRnm4).unsqueeze(-1) * mix_prod

        lap_er = (
            (bexp_er / (xyz * xyz).sum(-1)).unsqueeze(-1)
            * mix_prod
            * (bas_exp.unsqueeze(-1) + 1.0 / R.unsqueeze(-1))
        )

        return (
            lap_rn * er.unsqueeze(-1)
            + (
                nabla_rn[..., [[0, 1], [0, 2], [1, 2]]]
                * nabla_er[..., [[1, 0], [2, 0], [2, 1]]]
            ).sum(-1)
            + rn.unsqueeze(-1) * lap_er
        )

    # computes the basic quantities
    rn = fast_power(R, bas_n)
    er = torch.exp(-bas_exp * R)

    # computes the grad
    if any(x in derivative for x in [1, 2, 3]):
        Rnm2 = R ** (bas_n - 2)
        nRnm2 = bas_n * Rnm2
        bexp_er = bas_exp * er
        nabla_rn = (nRnm2).unsqueeze(-1) * xyz
        nabla_er = -(bexp_er).unsqueeze(-1) * xyz / R.unsqueeze(-1)

    return return_required_data(
        derivative,
        _kernel,
        _first_derivative_kernel,
        _second_derivative_kernel,
        _mixed_second_derivative_kernel,
    )


def radial_gaussian(
    R: torch.Tensor,  # distance between each electron and each atom
    bas_n: torch.Tensor,  # principal quantum number
    bas_exp: torch.Tensor,  # exponents of the exponential
    xyz: torch.Tensor = None,  # positions of the electrons
    derivative: list = [0],  # degree of the derivative
    sum_grad: bool = True,  # return the sum of the gradients
    sum_hess: bool = True,  # return the sum of the hessian
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute the radial part of GTOs (or its derivative).

    .. math::
        gto = r^n exp(-\alpha r^2)

    Args:
        R (torch.Tensor): distance between each electron and each atom
        bas_n (torch.Tensor): principal quantum number
        bas_exp (torch.Tensor): exponents of the exponential

    Keyword Args:
        xyz (torch.Tensor): positions of the electrons
                            (needed for derivative) (default: {None})
        derivative (list): degree of the derivative (default: {[0]})
                           0: value of the function
                           1: first derivative
                           2: pure second derivative
                           3: mixed second derivative
        sum_grad (bool): return the sum of the gradients (default: {True})
        sum_hess (bool): return the sum of the hessian (default: {True})

    Returns:
        torch.Tensor: values of each orbital radial part at each position
    """

    if not isinstance(derivative, list):
        derivative = [derivative]

    def _kernel() -> torch.Tensor:
        return rn * er

    def _first_derivative_kernel() -> torch.Tensor:
        if sum_grad:
            nabla_rn_sum = nabla_rn.sum(3)
            nabla_er_sum = nabla_er.sum(3)
            return nabla_rn_sum * er + rn * nabla_er_sum
        else:
            return nabla_rn * er.unsqueeze(-1) + rn.unsqueeze(-1) * nabla_er

    def _second_derivative_kernel() -> torch.Tensor:
        if sum_hess:
            lap_rn = nRnm2 * (bas_n + 1)
            lap_er = bas_exp * er * (4 * bas_exp * R2 - 6)

            return lap_rn * er + 2 * (nabla_rn * nabla_er).sum(3) + rn * lap_er

        else:
            xyz2 = xyz * xyz

            lap_er = (bas_exp * er).unsqueeze(-1) * (
                4 * bas_exp.unsqueeze(-1) * xyz2 - 2
            )

            xyz2 = xyz2 / xyz2.sum(-1, keepdim=True)

            lap_rn = nRnm2.unsqueeze(-1) * (1.0 + (bas_n - 2).unsqueeze(-1) * xyz2)

            return (
                lap_rn * er.unsqueeze(-1)
                + 2 * (nabla_rn * nabla_er)
                + rn.unsqueeze(-1) * lap_er
            )

    def _mixed_second_derivative_kernel() -> torch.Tensor:
        """Returns the mixed second derivative i.e. d^2/dxdy.
        where x and y are coordinate of the same electron."""

        mix_prod = xyz[..., [[0, 1], [0, 2], [1, 2]]].prod(-1)
        nRnm4 = nRnm2 / (xyz * xyz).sum(-1)

        lap_rn = ((bas_n - 2) * nRnm4).unsqueeze(-1) * mix_prod

        lap_er = 4 * (bexp_er * bas_exp).unsqueeze(-1) * mix_prod

        return (
            lap_rn * er.unsqueeze(-1)
            + (
                nabla_rn[..., [[0, 1], [0, 2], [1, 2]]]
                * nabla_er[..., [[1, 0], [2, 0], [2, 1]]]
            ).sum(-1)
            + rn.unsqueeze(-1) * lap_er
        )

    # computes the basic quantities
    R2 = R * R
    rn = fast_power(R, bas_n)
    er = torch.exp(-bas_exp * R2)

    # computes the grads
    if any(x in derivative for x in [1, 2, 3]):
        Rnm2 = R ** (bas_n - 2)
        nRnm2 = bas_n * Rnm2
        bexp_er = bas_exp * er

        nabla_rn = (nRnm2).unsqueeze(-1) * xyz
        nabla_er = -2 * (bexp_er).unsqueeze(-1) * xyz

    return return_required_data(
        derivative,
        _kernel,
        _first_derivative_kernel,
        _second_derivative_kernel,
        _mixed_second_derivative_kernel,
    )


def radial_gaussian_pure(
    R: torch.Tensor,  # distance between each electron and each atom
    bas_n: torch.Tensor,  # principal quantum number
    bas_exp: torch.Tensor,  # exponents of the exponential
    xyz: torch.Tensor = None,  # positions of the electrons
    derivative: List[int] = [0],  # degree of the derivative
    sum_grad: bool = True,  # return the sum_grad, i.e the sum of the gradients
    sum_hess: bool = True,  # return the sum_hess, i.e the sum of the lapacian
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute the radial part of GTOs (or its derivative).

    .. math:
        gto = exp(-\alpha r ^ 2)

    Args:
        R(torch.Tensor): distance between each electron and each atom
        bas_n(torch.Tensor): principal quantum number (not relevant here but kept for consistency)
        bas_exp(torch.Tensor): exponents of the exponential

    Keyword Arguments:
        xyz(torch.Tensor): positions of the electrons
                            (needed for derivative)(default: {None})
        derivative(int): degree of the derivative(default: {0})
        sum_grad(bool): return the sum_grad, i.e the sum of the gradients
                           (default: {True})
        sum_hess(bool): return the sum_hess, i.e the sum of the lapacian
                           (default: {True})

    Returns:
        torch.Tensor: values of each orbital radial part at each position
    """

    if not isinstance(derivative, list):
        derivative = [derivative]

    def _kernel():
        return er

    def _first_derivative_kernel():
        if sum_grad:
            return nabla_er.sum(3)
        else:
            return nabla_er

    def _second_derivative_kernel():
        if sum_hess:
            lap_er = bas_exp * er * (4 * bas_exp * R2 - 6)
            return lap_er
        else:
            xyz2 = xyz * xyz
            lap_er = (bas_exp * er).unsqueeze(-1) * (
                4 * bas_exp.unsqueeze(-1) * xyz2 - 2
            )
            return lap_er

    def _mixed_second_derivative_kernel():
        """Returns the mixed second derivative i.e. d^2/dxdy.
        where x and y are coordinate of the same electron."""

        mix_prod = xyz[..., [[0, 1], [0, 2], [1, 2]]].prod(-1)
        lap_er = 4 * (bexp_er * bas_exp).unsqueeze(-1) * mix_prod

        return lap_er

    # computes the basic  quantities
    R2 = R * R
    er = torch.exp(-bas_exp * R2)

    # computes the grads
    if any(x in derivative for x in [1, 2, 3]):
        bexp_er = bas_exp * er
        nabla_er = -2 * (bexp_er).unsqueeze(-1) * xyz

    return return_required_data(
        derivative,
        _kernel,
        _first_derivative_kernel,
        _second_derivative_kernel,
        _mixed_second_derivative_kernel,
    )


def radial_slater_pure(
    R: torch.Tensor,  # distance between each electron and each atom
    bas_n: torch.Tensor,  # principal quantum number
    bas_exp: torch.Tensor,  # exponents of the exponential
    xyz: torch.Tensor = None,  # positions of the electrons
    derivative: Union[int, List[int]] = 0,  # degree of the derivative
    sum_grad: bool = True,  # return the sum_grad, i.e the sum of the gradients
    sum_hess: bool = True,  # return the sum_hess, i.e the sum of the laplacian
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute the radial part of STOs (or its derivative).

    .. math::
        sto = exp(-\alpha | r |)

    Args:
        R (torch.Tensor): distance between each electron and each atom
        bas_n (torch.Tensor): principal quantum number (not relevant here but kept for consistency)
        bas_exp (torch.Tensor): exponents of the exponential

    Keyword Arguments:
        xyz (torch.Tensor): positions of the electrons
                            (needed for derivative)(default: {None})
        derivative (Union[int, List[int]]): degree of the derivative(default: {0})
        sum_grad (bool): return the sum_grad, i.e the sum of the gradients
                           (default: {True})
        sum_hess (bool): return the sum_hess, i.e the sum of the laplacian
                           (default: {True})

    Returns:
        torch.Tensor: values of each orbital radial part at each position
    """

    if not isinstance(derivative, list):
        derivative = [derivative]

    def _kernel() -> torch.Tensor:
        return er

    def _first_derivative_kernel() -> torch.Tensor:
        if sum_grad:
            return nabla_er.sum(3)
        else:
            return nabla_er

    def _second_derivative_kernel() -> torch.Tensor:
        if sum_hess:
            return bexp_er * (bas_exp - 2.0 / R)

        else:
            xyz2 = xyz * xyz / (R * R).unsqueeze(-1)
            lap_er = bexp_er.unsqueeze(-1) * (
                bas_exp.unsqueeze(-1) * xyz2 - (1 - xyz2) / R.unsqueeze(-1)
            )
            return lap_er

    def _mixed_second_derivative_kernel() -> torch.Tensor:
        """Returns the mixed second derivative i.e. d^2/dxdy.
        where x and y are coordinate of the same electron."""

        mix_prod = xyz[..., [[0, 1], [0, 2], [1, 2]]].prod(-1)

        lap_er = (
            (bexp_er / (xyz * xyz).sum(-1)).unsqueeze(-1)
            * mix_prod
            * (bas_exp.unsqueeze(-1) + 1.0 / R.unsqueeze(-1))
        )

        return lap_er

    # computes the basic gradients
    er = torch.exp(-bas_exp * R)

    # computes the grad
    if any(x in derivative for x in [1, 2, 3]):
        bexp_er = bas_exp * er
        nabla_er = -(bexp_er).unsqueeze(-1) * xyz / R.unsqueeze(-1)

    return return_required_data(
        derivative,
        _kernel,
        _first_derivative_kernel,
        _second_derivative_kernel,
        _mixed_second_derivative_kernel,
    )


def return_required_data(
    derivative: List[int],
    _kernel: Callable,
    _first_derivative_kernel: Callable,
    _second_derivative_kernel: Callable,
    _mixed_second_derivative_kernel: Callable,
) -> Union[List, torch.Tensor]:
    """Returns the data contained in derivative

    Args:
        derivative (List[int]): list of the derivatives required
        _kernel (Callable): kernel of the values
        _first_derivative_kernel (Callable): kernel for 1st der
        _second_derivative_kernel (Callable): kernel for 2nd der

    Returns:
        Union[List, torch.Tensor]: values of the different der required
    """

    # prepare the output/kernel
    output: List = []
    fns: List[Callable] = [
        _kernel,
        _first_derivative_kernel,
        _second_derivative_kernel,
        _mixed_second_derivative_kernel,
    ]

    # compute the requested functions
    for d in derivative:
        output.append(fns[d]())

    if len(derivative) == 1:
        return output[0]
    else:
        return output
