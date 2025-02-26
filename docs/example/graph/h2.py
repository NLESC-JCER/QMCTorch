from qmctorch.scf import Molecule
from qmctorch.wavefunction.slater_jastrow import SlaterJastrow
from qmctorch.wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel
from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver
from qmctorch.utils import set_torch_double_precision
from qmctorch.wavefunction.jastrows.graph.mgcn_jastrow import MGCNJastrowFactor
set_torch_double_precision()

# define the molecule
mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69',
               calculator='pyscf', basis='dzp', unit='bohr')

# jastrow
jastrow = MGCNJastrowFactor(
            mol,
            ee_model_kwargs={"n_layers": 3, "feats": 32, "cutoff": 5.0, "gap": 1.0},
            en_model_kwargs={"n_layers": 3, "feats": 32, "cutoff": 5.0, "gap": 1.0},
        )


# define the wave function
wf = SlaterJastrow(mol, kinetic='jacobi',
                   configs='ground_state', jastrow=jastrow) #.gto2sto()

# sampler
sampler = Metropolis(nwalkers=1000, nstep=1000, step_size=0.25,
                     nelec=wf.nelec, ndim=wf.ndim,
                     init=mol.domain('atomic'),
                     move={'type': 'one-elec', 'proba': 'normal'},
                     logspace=False)


# solver
solver = Solver(wf=wf, sampler=sampler)

# single point
obs = solver.single_point()