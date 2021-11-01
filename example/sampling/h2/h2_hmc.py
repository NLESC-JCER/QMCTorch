import torch
from torch import optim
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

from qmctorch.sampler import Metropolis, Hamiltonian
from qmctorch.scf import Molecule
from qmctorch.solver import SolverSlaterJastrow
from qmctorch.utils import plot_energy, plot_walkers_traj, set_torch_double_precision
from qmctorch.wavefunction import SlaterJastrowBackFlow, SlaterJastrow

cuda = torch.cuda.is_available()

torch.manual_seed(0)
set_torch_double_precision()

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


if config.calculator == 'adf':
    mol = Molecule(load='H2_adf_CVB3.hdf5')
    wf = SlaterJastrow(mol, configs='single_double(2,2)', cuda=cuda)
else:
    mol = Molecule(atom='h2.xyz',
                   unit='angs',
                   calculator=config.calculator,
                   basis=config.basis_set,
                   name='H2')
    wf = SlaterJastrow(mol, configs='single_double(2,2)', cuda=cuda).gto2sto()

opt = optim.Adam(wf.parameters(), lr=config.lr)

sampler = Hamiltonian(
    nwalkers=config.nwalkers,
    nstep=config.nstep_hmc,
    ntherm=config.ntherm_hmc,
    ndecor=config.ndecor_hmc,
    step_size=config.step_size_hmc,
    L=config.L_hmc,
    ndim=wf.ndim,
    nelec=wf.nelec,
    init=mol.domain('atomic'),
    cuda=cuda)

# step_sizes = np.logspace(-1.5, -0.5, 30)
#
# M = 50
# tau_ints = []
#
# for s in step_sizes:
#     sampler.step_size = s
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
                                            'nstep_update': config.nstep_update_hmc})
obs = solver.run(config.nepoch, batchsize=sampler.nwalkers)

# plot_energy(obs.local_energy)
# plot_energy(obs.energy)

np.save('H2_HMC.npy', obs.local_energy)
