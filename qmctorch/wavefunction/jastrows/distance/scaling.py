import torch


def get_scaled_distance(kappa, r):
    """compute the scaled distance
    .. math::
        u = \frac{1+e^{-kr}}{k}

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

    Args:
        kappa (float) : scaling factor
        r (torch.tensor): unsqueezed matrix of the e-e distances
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
