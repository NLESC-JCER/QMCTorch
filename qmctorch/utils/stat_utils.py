import numpy as np
from scipy.optimize import curve_fit


def blocking(x, block_size, expand=False):
    """block the data

    Args:
        x (data): size Nsample, Nexp
        block_size (int): size of the block
    """
    nstep, nwalkers = x.shape
    nblock = nstep // block_size

    xb = np.copy(x[:block_size * nblock, :])
    xb = xb.reshape(nblock, block_size, nwalkers).mean(axis=1)

    if expand:
        xb = xb.T.repeat(block_size).reshape(nwalkers, -1).T[:nstep]

    return xb


def correlation_coefficient(x, norm=True):
    """Computes the correlation coefficient

    Args:
        x (np.ndarray): measurement of size [Nsample, Nexperiments]
        norm (bool, optional): [description]. Defaults to True.
    """

    N = x.shape[0]
    xm = x-x.mean(0)

    c = np.zeros_like(x)
    for tau in range(0, N):
        c[tau] = 1./(N-tau) * (xm[:N-tau] * xm[tau:]).sum(0)

    if norm:
        c /= c[0]

    return c


def integrated_autocorrelation_time(correlation_coeff, size_max):
    """Computes the integrated autocorrelation time

    Args:
        correlation_coeff (np.ndarray): coeff size Nsample,Nexp
        size_max (int): max size 
    """
    return 1. + 2. * np.cumsum(correlation_coeff[1:size_max], 0)


def fit_correlation_coefficient(coeff):
    """Fit the correlation coefficient 
       to get the correlation time.

    Args:
        coeff (np.ndarray): correlation coefficient

    Returns:
        float, np.ndarray: correlation time, fitted curve
    """

    def fit_exp(x, y):
        """Fit an exponential to the data."""
        def func(x, tau):
            return np.exp(-x/tau)
        popt, pcov = curve_fit(func, x, y, p0=(1.))
        return popt[0], func(x, popt)

    return fit_exp(np.arange(len(coeff)), coeff)
