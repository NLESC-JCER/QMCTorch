import torch
from torch import optim
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

from qmctorch.sampler import Metropolis, Hamiltonian
from qmctorch.scf import Molecule
from qmctorch.solver import SolverSlaterJastrow
from qmctorch.utils import plot_energy, plot_walkers_traj
from qmctorch.wavefunction import SlaterJastrowBackFlow, SlaterJastrow

torch.manual_seed(0)

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def correlation_coefficient(x, norm=True):
    """Computes the correlation coefficient using the FFT

    Args:
        x (np.ndarray): measurement of size [MC steps, N walkers]
        norm (bool, optional): [description]. Defaults to True.
    """

    N = x.shape[0]
    xm = x - x.mean(0)

    c = fftconvolve(xm, xm[::-1], axes=0)[N - 1:]

    if norm:
        c /= c[0]

    return c


# molecule
mol = Molecule(atom='h2.xyz',
               unit='angs',
               calculator=config.calculator,
               basis='sto-3g',
               name='H2')

wf = SlaterJastrow(mol, configs='ground_state')
# wf = SlaterJastrowBackFlow(mol, configs='single(2,2)')

opt = optim.Adam(wf.parameters(), lr=config.lr)

sampler = Metropolis(
    nwalkers=config.nwalkers,
    nstep=config.nstep_m,
    ntherm=-1,
    ndecor=3,
    step_size=config.step_size_m,
    ndim=wf.ndim,
    nelec=wf.nelec,
    init=mol.domain('atomic'),
    move={
        'type': 'all-elec',
        'proba': 'normal'},
    cuda=False)

# step_sizes = np.logspace(-1, 0.5, 30)
#
# M = 100
# tau_ints = []
#
# for s in step_sizes:
#     sampler.step_size = s
#     sampler.configure_move(move={'type': 'all-elec', 'proba': 'normal'})
#     rho = correlation_coefficient(wf.pdf(sampler(wf.pdf)).detach().numpy())
#     tau_int = 1 + 2 * np.sum(rho[1:M], 0).mean()
#     tau_ints.append(tau_int)
#
# plt.plot(step_sizes, tau_ints)
# plt.xscale('log')
# plt.show()

solver = SolverSlaterJastrow(wf=wf, sampler=sampler, optimizer=opt)
solver.configure(grad='manual', resampling={'mode': 'update',
                                            'resample_every': 1,
                                            'nstep_update': config.nstep_update_m})
obs = solver.run(config.nepoch, batchsize=sampler.nwalkers)

plot_energy(obs.local_energy)
# plot_energy(obs.energy)

np.save('H2_M.npy', obs.local_energy)
