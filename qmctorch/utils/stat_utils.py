import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve


def blocking(x, block_size, expand=False):
    """block the data

    Args:
        x (data): size Nsample, Nexp
        block_size (int): size of the block
    """
    nstep, nwalkers = x.shape
    nblock = nstep // block_size

    xb = np.copy(x[: block_size * nblock, :])
    xb = xb.reshape(nblock, block_size, nwalkers).mean(axis=1)

    if expand:
        xb = xb.T.repeat(block_size).reshape(nwalkers, -1).T[:nstep]

    return xb


def correlation_coefficient(x, norm=True):
    """Computes the correlation coefficient using the FFT

    Args:
        x (np.ndarray): measurement of size [MC steps, N walkers]
        norm (bool, optional): [description]. Defaults to True.
    """

    N = x.shape[0]
    xm = x - x.mean(0)

    c = fftconvolve(xm, xm[::-1], axes=0)[N - 1 :]

    if norm:
        c /= c[0]

    return c


def integrated_autocorrelation_time(correlation_coeff, size_max):
    """Computes the integrated autocorrelation time

    Args:
        correlation_coeff (np.ndarray): coeff size Nsample,Nexp
        size_max (int): max size
    """
    return 1.0 + 2.0 * np.cumsum(correlation_coeff[1:size_max], 0)


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
            return np.exp(-x / tau)

        popt, _ = curve_fit(func, x, y, p0=(1.0))
        return popt[0], func(x, popt)

    return fit_exp(np.arange(len(coeff)), coeff)
