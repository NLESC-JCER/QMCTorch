import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_observable(obs_dict,e0=None,ax=None):
    '''Plot the observable selected.

    Args:
        obs_dict : dictioanry of observable
    '''
    show_plot = False
    if ax is None:    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        show_plot = True

    n = len(obs_dict['local_energy'])
    epoch = np.arange(n)

    # get the variance
    emax = [np.quantile(e,0.75) for e in obs_dict['local_energy'] ]
    emin = [np.quantile(e,0.25) for e in obs_dict['local_energy'] ]

    # get the mean value
    energy = np.mean(obs_dict['local_energy'],1)

    # plot
    ax.fill_between(epoch,emin,emax,alpha=0.5,color='#4298f4')
    ax.plot(epoch,energy,color='#144477')
    if e0 is not None:
        ax.axhline(e0,color='black',linestyle='--')

    ax.grid()
    ax.set_xlabel('Number of epoch')
    ax.set_ylabel('Energy')

    if show_plot:
        plt.show()