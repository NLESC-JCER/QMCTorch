import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_observable(obs_dict, e0=None, ax=None, var=False):
    '''Plot the observable selected.

    Args:
        obs_dict : dictioanry of observable
    '''

    show_plot = False

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        show_plot = True

    if isinstance(obs_dict, dict):
        data = obs_dict['local_energy']
    else:
        data = np.hstack(np.squeeze(np.array(obs_dict)))

    n = len(data)
    epoch = np.arange(n)

    # get the variance
    energy = np.array([np.mean(e) for e in data])
    variance = np.array([np.var(e) for e in data])
    emax = [np.quantile(e, 0.75) for e in data]
    emin = [np.quantile(e, 0.25) for e in data]

    # get the mean value
    print("Energy   : %f " % np.mean(energy))
    print("Variance : %f " % np.std(energy))

    # plot
    # ax.fill_between(epoch, emin, emax, alpha=0.5, color='#4298f4')
    ax.fill_between(epoch, energy-variance, energy +
                    variance, alpha=0.5, color='#4298f4')
    ax.plot(epoch, energy, color='#144477')
    if e0 is not None:
        ax.axhline(e0, color='black', linestyle='--')

    ax.grid()
    ax.set_xlabel('Number of epoch')
    ax.set_ylabel('Energy', color='black')

    if var:
        ax2 = ax.twinx()
        ax2.plot(epoch, variance, color='blue')
        ax2.set_ylabel('variance', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        fig.tight_layout()

    if show_plot:
        plt.show()


def plot_block(obs_dict):

    eloc = np.array(obs_dict['local_energy']).squeeze(-1)
    nstep, nwalkers = eloc.shape
    block_size_max = nstep//2

    evar = []
    for size in range(1, block_size_max):
        nblock = nstep//size
        tmp = np.copy(eloc[:size*nblock, :])
        tmp = tmp.reshape(size, nblock, nwalkers).mean(axis=0)
        evar.append(np.sqrt(np.var(tmp)/ (nblock-1)) )
    np.savetxt('block.dat',np.array(evar))
    print(evar)
    plt.plot(np.array(evar))
    plt.show()

def plot_block_baby(obs_dict):

    
    m = np.array(obs_dict['local_energy'])
    K = m.shape[0]
    Keq = 0
    Keff = K - Keq
    nstepx = Keff//2
    err = []
    for i in range(1,nstepx):
        nblock = Keff//i
        iter = 0
        mcum=0
        mcm2=0
        for j in range(1,nblock):
            msum=0
            for l in range(1,i):
                iter=iter+1
                msum=msum+m[iter+Keq]
        
        mcum=mcum+msum/i
        mcm2=mcm2+msum*msum/i**2
    
    mcum=mcum/nblock;
    mcm2=mcm2/nblock;
    err.append(np.sqrt((mcm2-mcum**2)/(nblock-1)))
    print(err)
    np.savetxt('block.dat',np.array(err))

def save_observalbe(filename, obs_dict):
    with open(filename, 'wb') as fhandle:
        pickle.dump(obs_dict, fhandle, protocol=pickle.HIGHEST_PROTOCOL)


def load_observable(filename):
    with open(filename, 'rb') as fhandle:
        return pickle.load(fhandle)
