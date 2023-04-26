import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from .stat_utils import (blocking, correlation_coefficient,
                         fit_correlation_coefficient,
                         integrated_autocorrelation_time)


def plot_energy(local_energy, e0=None, show_variance=False):
    """Plot the evolution of the energy

    Args:
        local_energy (np.ndarray): local energies along the trajectory
        e0 (float, optional): Target value for the energy. Defaults to None.
        show_variance (bool, optional): show the variance if True. Defaults to False.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    n = len(local_energy)
    epoch = np.arange(n)

    # get the variance
    energy = np.array([np.mean(e) for e in local_energy])
    variance = np.array([np.var(e) for e in local_energy])

    # plot
    ax.fill_between(epoch, energy - variance, energy +
                    variance, alpha=0.5, color='#4298f4')
    ax.plot(epoch, energy, color='#144477')
    if e0 is not None:
        ax.axhline(e0, color='black', linestyle='--')

    ax.grid()
    ax.set_xlabel('Number of epoch')
    ax.set_ylabel('Energy', color='black')

    if show_variance:
        ax2 = ax.twinx()
        ax2.plot(epoch, variance, color='blue')
        ax2.set_ylabel('variance', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        fig.tight_layout()

    plt.show()


def plot_data(observable, obsname):
    """Plot the evolution a given data

    Args:
        obs_dict (SimpleNamespace): namespace of observable
        obsname (str): name (key) of the desired observable
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    data = np.array(observable.__dict__[obsname]).squeeze()
    epoch = np.arange(len(data))
    ax.plot(epoch, data, color='#144477')

    plt.show()


def plot_walkers_traj(eloc, walkers='mean'):
    """Plot the trajectory of all the individual walkers

    Args:
        obs (SimpleNamespace): Namespace of the observables
        walkers (int, str, optional): all, mean or index of a given walker Defaults to 'all'
    """
    nstep, nwalkers = eloc.shape
    celoc = np.cumsum(eloc, axis=0).T
    celoc /= np.arange(1, nstep + 1)

    if walkers is not None:

        if walkers == 'all':
            plt.plot(eloc, 'o', alpha=1 / nwalkers, c='grey')
            cmap = cm.hot(np.linspace(0, 1, nwalkers))
            for i in range(nwalkers):
                plt.plot(celoc.T[:, i], color=cmap[i])

        elif walkers == 'mean':
            plt.plot(eloc, 'o', alpha=1 / nwalkers, c='grey')
            emean = np.mean(celoc.T, axis=1)
            emin = emean.min()
            emax = emean.max()
            delta = emax-emin
            plt.plot(emean, linewidth=5)
            plt.ylim(emin-0.25*delta,emax+0.25*delta)
        else:
            raise ValueError('walkers argument must be all or mean')
        
        plt.grid()
        plt.xlabel('Monte Carlo Steps')
        plt.ylabel('Energy (Hartree)')

    plt.show()


def plot_correlation_coefficient(eloc, size_max=100):
    """Plot the correlation coefficient of the local energy
       and fit the curve to an exp to extract the correlation time.

    Args:
        eloc (np.ndarray): values of the local energy (Nstep, Nwalk)
        size_max (int, optional): maximu number of MC step to consider.Defaults to 100.

    Returns:
        np.ndarray, float: correlation coefficients (size_max, Nwalkers), correlation time
    """

    rho = correlation_coefficient(eloc)

    tau_fit, fitted = fit_correlation_coefficient(
        rho.mean(1)[:size_max])

    plt.plot(rho, alpha=0.25)
    plt.plot(rho.mean(1), linewidth=3, c='black')
    plt.plot(fitted, '--', c='grey')
    plt.xlim([0, size_max])
    plt.ylim([-0.25, 1.5])
    plt.xlabel('MC steps')
    plt.ylabel('Correlation coefficient')
    plt.text(0.5*size_max, 1.05, 'tau=%1.3f' %
             tau_fit, {'color': 'black',  'fontsize': 15})
    plt.grid()
    plt.show()

    return rho, tau_fit


def plot_integrated_autocorrelation_time(eloc, rho=None, size_max=100, C=5):
    """compute/plot the integrated autocorrelation time

    Args:
        eloc (np.ndarray, optional): local energy values (Nstep, Nwalkers)
        rho (np.ndarray, optional): Correlation coefficient. Defaults to None.
        size_max (int, optional): maximu number of MC step to consider.Defaults to 100.
        C (int, optional): [description]. Defaults to 5.
    """

    if rho is None:
        rho = correlation_coefficient(eloc)

    tau = integrated_autocorrelation_time(rho, size_max)

    tc, idx_tc = [], []
    idx = np.arange(1, size_max)
    for iw in range(eloc.shape[1]):

        t = tau[:, iw]
        if len(t[C*t <= idx]) > 0:

            tval = t[C*t <= idx][0]
            ii = np.where(t == tval)[0][0]

            tc.append(tval)
            idx_tc.append(ii)

    plt.plot(tau, alpha=0.25)
    tm = tau.mean(1)
    plt.plot(tm, c='black')
    plt.plot(idx/C, '--', c='grey')

    plt.plot(idx_tc, tc, 'o', alpha=0.25)
    tt = tm[tm*C <= idx][0]
    ii = np.where(tm == tt)[0][0]
    plt.plot(ii, tt, 'o')

    plt.grid()
    plt.xlabel('MC step')
    plt.ylabel('IAC')
    plt.show()


def plot_blocking_energy(eloc, block_size, walkers='mean'):
    """Plot the blocked energy values

    Args:
        eloc (np.ndarray): values of the local energies
        block_size (int): size of the block
        walkers (str, optional): which walkers to plot (mean, all, index or list). Defaults to 'mean'.

    Raises:
        ValueError: [description]
    """
    eb = blocking(eloc, block_size, expand=True)
    if walkers == 'all':
        plt.plot(eloc)
        plt.plot(eb)

    elif walkers == 'mean':
        plt.plot(eloc.mean(1))
        plt.plot(eb.mean(1))

    elif walkers.__class__.__name__ in ['int', 'list']:
        plt.plot(eloc[:, walkers])
        plt.plot(eb[:, walkers])

    else:
        raise ValueError('walkers ', walkers, ' not recognized')

    plt.grid()
    plt.xlabel('MC steps')
    plt.ylabel('Energy')
    plt.show()

    return blocking(eloc, block_size, expand=False)


def plot_correlation_time(eloc):
    """Plot the blocking thingy

    Args:
        eloc (np.array): values of the local energy
    """

    nstep, nwalkers = eloc.shape
    max_block_size = nstep // 2

    var = np.std(eloc, axis=0)

    evar = []
    for size in range(1, max_block_size):
        eb = blocking(eloc, size)
        evar.append(np.std(eb, axis=0) * size / var)

    plt.plot(np.array(evar))
    plt.xlabel('Blocking size')
    plt.ylabel('Correlation steps')
    plt.show()


def plot_block(eloc):
    """Plot the blocking thingy

    Args:
        eloc (np.array): values of the local energy
    """

    nstep, nwalkers = eloc.shape
    max_block_size = nstep // 2

    evar = []
    for size in range(1, max_block_size):
        eb = blocking(eloc, size)
        nblock = eb.shape[0]
        evar.append(np.sqrt(np.var(eb, axis=0) / (nblock - 1)))

    plt.plot(np.array(evar))
    plt.show()
