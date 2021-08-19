import torch
from torch import optim

from qmctorch.sampler import Metropolis
from qmctorch.scf import Molecule
from qmctorch.solver import SolverSlaterJastrow
from qmctorch.utils import plot_correlation_coefficient, plot_integrated_autocorrelation_time
from qmctorch.wavefunction.slater_jastrow_unified import SlaterJastrowUnified as SlaterJastrow
from qmctorch.wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel
torch.manual_seed(0)

# molecule
mol = Molecule(
    atom='H 0 0 -0.69; H 0 0 0.69',
    unit='bohr',
    calculator='pyscf',
    basis='sto-3g')


# jastrow
jastrow = JastrowFactor(mol, PadeJastrowKernel)

# wave funtion
wf = SlaterJastrow(mol, kinetic='auto',
                   jastrow=jastrow,
                   configs='single(2,2)')

# sampler
sampler = Metropolis(
    nwalkers=10,
    nstep=1000,
    ntherm=0,
    ndecor=1,
    step_size=0.5,
    ndim=wf.ndim,
    nelec=wf.nelec,
    init=mol.domain('normal'),
    move={
        'type': 'all-elec',
        'proba': 'normal'})

opt = optim.Adam(wf.parameters(), lr=0.01)

solver = SolverSlaterJastrow(wf=wf, sampler=sampler, optimizer=opt)

pos = solver.sampler(wf.pdf)
obs = solver.sampling_traj(pos)

rho, tau = plot_correlation_coefficient(obs.local_energy)
print(f'fit exp(-x/tau), tau={tau}')
iat = plot_integrated_autocorrelation_time(
    obs.local_energy, rho=rho, C=5)
print(f"integrated autocorrelation time: {iat}")
