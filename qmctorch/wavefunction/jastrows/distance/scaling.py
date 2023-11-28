import torch


def get_scaled_distance(kappa, r):
    """compute the scaled distance

    .. math::
        u_{ij} = \\frac{1-e^{-\\kappa r_{ij}}}{\\kappa}

    Args:
        kappa (float) : scaling factor
        r (torch.tensor): matrix of the e-e distances
                            Nbatch x Nelec x Nelec

    Returns:
        torch.tensor: values of the scaled distance
                        Nbatch, Nelec, Nelec
    """
    return (1. - torch.exp(-kappa * r))/kappa


def get_der_scaled_distance(kappa, r, dr):
    """Returns the derivative of the scaled distances

    .. math::
        \\frac{d u}{d x_i} = \\frac{d r_{ij}}{d x_i} e^{-\\kappa r_{ij}}
    Args:
        kappa (float) : scaling factor
        r (torch.tensor): matrix of the e-e distances
                            Nbatch x Nelec x Nelec

        dr (torch.tensor): matrix of the derivative of the e-e distances
                            Nbatch x Ndim x Nelec x Nelec

    Returns:
        torch.tensor : deriative of the scaled distance
                        Nbatch x Ndim x Nelec x Nelec
    """
    return dr * torch.exp(-kappa * r.unsqueeze(1))


def get_second_der_scaled_distance(kappa, r, dr, d2r):
    """computes the second derivative of the scaled distances

    .. math::
        \\frac{d^2u_{ij}}{d x_i^2} = \\frac{d^2r_{ij}}{d x_i^2} -\\kappa \\left( \\frac{d r_{ij}}{d x_i} \\right)^2 e^{-\\kappa r_{ij}}

    Args:
        kappa (float) : scaling factor
        r (torch.tensor): unsqueezed matrix of the e-e distances
                            Nbatch x Nelec x Nelec
        dr (torch.tensor): matrix of the derivative of the e-e distances
                            Nbatch x Ndim x Nelec x Nelec
        d2r (torch.tensor): matrix of the 2nd derivative of
                            the e-e distances
                            Nbatch x Ndim x Nelec x Nelec

    Returns:
        torch.tensor : second deriative of the scaled distance
                        Nbatch x Ndim x Nelec x Nelec
    """
    return (d2r - kappa * dr * dr) * torch.exp(-kappa*r.unsqueeze(1))
