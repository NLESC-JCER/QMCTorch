import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle

from .stat_utils import blocking, correlation_coefficient, integrated_autocorrelation_time


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

    # get the mean value
    print("Energy   : %f " % np.mean(energy))
    print("Variance : %f " % np.std(energy))

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

    data = np.array(observable.__getattribute__(obsname)).flatten()
    epoch = np.arange(len(data))
    ax.plot(epoch, data, color='#144477')

    if obsname + '.grad' in observable.__dict__.keys():
        data = np.array(observable.__getattribute__(
            obsname + '.grad')).flatten()
        ax2 = ax.twinx()
        ax2.plot(epoch, data, color='blue')
        ax2.set_ylabel('gradient', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

    plt.show()


def plot_walkers_traj(eloc, walkers='mean'):
    """Plot the trajectory of all the individual walkers

    Args:
        obs (SimpleNamespace): Namespace of the observables
        walkers (int, str, optional): all, mean or index of a given walker Defaults to 'all'

    Returns:
        float :  Decorelation time
    """

    # eloc = np.array(obs.local_energy).squeeze(-1)

    nstep, nwalkers = eloc.shape

    celoc = np.cumsum(eloc, axis=0).T
    celoc /= np.arange(1, nstep + 1)

    var_decor = np.sqrt(np.var(np.mean(celoc, axis=1)))
    var = np.sqrt(np.var(celoc, axis=1) / (nstep - 1))

    Tc = (var_decor / var)**2

    if walkers is not None:
        # plt.subplot(1, 2, 1)

        if walkers == 'all':
            plt.plot(eloc, 'o', alpha=1 / nwalkers, c='grey')
            cmap = cm.hot(np.linspace(0, 1, nwalkers))
            for i in range(nwalkers):
                plt.plot(celoc.T[:, i], color=cmap[i])

        elif walkers == 'mean':
            plt.plot(eloc, 'o', alpha=1 / nwalkers, c='grey')
            plt.plot(np.mean(celoc.T, axis=1), linewidth=5)

        else:
            plt.plot(eloc[walkers, :], 'o',
                     alpha=1 / nwalkers, c='grey')
            plt.plot(celoc.T[traj_index, :])
        plt.grid()
        plt.xlabel('Monte Carlo Steps')
        plt.ylabel('Energy (Hartree)')
        # plt.subplot(1, 2, 2)
        # plt.hist(Tc)

    plt.show()

    return Tc


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
        evar.append(np.sqrt(np.var(eb, axis=0) / (nblock - 1)))

    plt.plot(np.array(evar))
    plt.show()


def plot_autocorrelation(eloc, size_max=100, C=5):

    rho = correlation_coefficient(eloc)
    plt.plot(rho)
    plt.show()

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

    plt.plot(tau)
    plt.plot(idx_tc, tc, 'o')
    plt.plot(idx/5, '--', c='black')
    plt.show()

    tm = tau.mean(1)
    idx = np.arange(0, len(tm))
    plt.plot(idx/5, '--', c='black')
    tt = tm[tm*C <= idx][0]
    ii = np.where(tm == tt)[0][0]
    plt.plot(tau.mean(1))
    plt.plot(ii, tt, 'o')
    plt.plot(idx/5, '--', c='black')
    plt.show()
    return tau
