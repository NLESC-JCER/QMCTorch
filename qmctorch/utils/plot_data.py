import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle


def plot_energy(local_energy, e0=None, show_variance=False):
    """Plot the evolution of the energy along the trajectory

    Arguments:
        local_energy {np.array} -- array containing the local energy

    Keyword Arguments:
        e0 {float} -- optimal value of the enrgy (default: {None})
        show_variance {bool} -- plot the variance (default: {False})

    Returns:
        tuple -- (energy, variance)
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

    return energy, variance


def plot_data(observable, obsname):
    """Plot the evolution a given data

    Arguments:
        obs_dict {SimpleNamespace} -- namespace of observable
        obsname {str} -- name (key) of the desired observable
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


def plot_walkers_traj(obs, traj_index='all'):
    """Plot the trajectory of all the individual walkers

    Arguments:
        obs {SimpleNamespace} -- Namespace of the observables

    Keyword Arguments:
        traj_index {str} -- all or index of a given walker (default: {'all'})

    Returns:
        float -- decorelation time
    """

    eloc = np.array(obs.local_energy).squeeze(-1)

    nstep, nwalkers = eloc.shape

    celoc = np.cumsum(eloc, axis=0).T
    celoc /= np.arange(1, nstep + 1)

    var_decor = np.sqrt(np.var(np.mean(celoc, axis=1)))
    var = np.sqrt(np.var(celoc, axis=1) / (nstep - 1))

    Tc = (var_decor / var)**2

    if traj_index is not None:
        plt.subplot(1, 2, 1)

        if traj_index == 'all':

            plt.plot(eloc, 'o', alpha=1 / nwalkers, c='grey')
            cmap = cm.hot(np.linspace(0, 1, nwalkers))
            for i in range(nwalkers):
                plt.plot(celoc.T[:, i], color=cmap[i])
        else:
            plt.plot(eloc[traj_index, :], 'o',
                     alpha=1 / nwalkers, c='grey')
            plt.plot(celoc.T[traj_index, :])
        plt.subplot(1, 2, 2)
        plt.hist(Tc)

    plt.show()

    return Tc


def plot_block(eloc):
    """Plot the blocking thingy

    Arguments:
        eloc {np.array} -- values of the local energy
    """

    eloc = np.array(obs_dict['local_energy']).squeeze(-1)
    nstep, nwalkers = eloc.shape
    max_block_size = nstep // 2

    evar = []
    for size in range(1, max_block_size):
        nblock = nstep // size
        tmp = np.copy(eloc[:size * nblock, :])
        tmp = tmp.reshape(nblock, size, nwalkers).mean(axis=1)
        evar.append(np.sqrt(np.var(tmp, axis=0) / (nblock - 1)))

    plt.plot(np.array(evar))
    plt.show()
