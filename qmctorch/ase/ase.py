from ase.calculators.calculator import Calculator, all_changes
import torch
from torch import optim

from ..utils import set_torch_double_precision
from ..scf.molecule import Molecule
from ..wavefunction.slater_jastrow import SlaterJastrow
from ..wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel
from ..solver import Solver
from ..sampler import Metropolis

class QMCTorchCalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(self,
                 restart=None,
                 *,
                 labels=None,
                 atoms=None,
                 solver=None,
                 **kwargs):
        
        Calculator.__init__(self, restart=restart, labels=labels, atoms=atoms)
        self.use_cuda = torch.cuda.is_available()
        set_torch_double_precision()

    def set(self, **kwargs):
        raise NotImplementedError("Not done yet")

    def set_atoms(self, atoms):
        """
        Set atoms object.

        Parameters
        ----------
        atoms : ASE Atoms object
            The atoms object to be set.
        """
        self.atoms = atoms


    def reset(self):
        """
        Reset the calculator.
        """
        self.atoms = None
        self.reset_results()

    def reset_results(self):
        self.results = {}

    def calculate(self, atoms=None, properties=['energy'] ,system_changes=None):
        """_summary_

        Args:
            atoms (_type_, optional): _description_. Defaults to None.
            properties (list, optional): _description_. Defaults to ['energy'].
            system_changes (_type_, optional): _description_. Defaults to None.
        """

        if any([p not in properties for p in self.implemented_properties]):
            raise ValueError('property not recognized')
        
        for p in properties:
            if p is 'energy':
                self.calculate_energy(atoms=atoms)
            if p is 'froces':
                self.calculate_forces(atoms=atoms)


    def set_solver(self, atoms):
        """_summary_

        Args:
            atoms (_type_): _description_
        """
        xyz_filename = './mol.xyz'
        atoms.write(xyz_filename)

        mol = Molecule(atom=xyz_filename, unit='angs', calculator='adf', basis='dzp')
        jastrow = JastrowFactor(mol, PadeJastrowKernel, kernel_kwargs={'w':1.00}, cuda=self.use_gpu)
        configs =  'single_double(2,2)'

        wf = SlaterJastrow(mol, kinetic='jacobi',
                    configs=configs,
                    backflow=None,
                    jastrow=jastrow,
                    orthogonalize_mo=True,
                    cuda=self.use_gpu)
        
        sampler = Metropolis(nwalkers=4000, nstep=2000, nelec=wf.nelec, ntherm=-1, ndecor=1,
                    step_size=0.05, init=mol.domain('atomic'), cuda=self.use_gpu)

        lr_dict = [{'params': wf.jastrow.parameters(), 'lr': 1E-2},
                {'params': wf.ao.parameters(), 'lr': 1E-2},
                {'params': wf.mo.parameters(), 'lr': 1E-2},
                {'params': wf.fc.parameters(), 'lr': 1E-2}]
        opt = optim.Adam(lr_dict, lr=1E-2)


        solver = Solver(wf=wf, sampler=sampler, optimizer=opt, scheduler=None)
        solver.set_params_requires_grad(wf_params=True, geo_params=False)

        solver.configure(track=['local_energy', 'parameters'], freeze=[],
                loss='energy', grad='manual',
                ortho_mo=False, clip_loss=False,
                resampling={'mode': 'update','resample_every':1, 'nstep_update':50, 'ntherm_update':-1}
                )
        return solver

    def calculate_energy(self, atoms=None):
        """_summary_

        Args:
            atoms (_type_, optional): _description_. Defaults to None.
        """
        Calculator.calculate(self, atoms)
        atoms = self.atoms
        solver = self.set_solver(atoms)
        solver.run(5, tqdm=True)

    def calculate_forces(self, atoms, d=0.001):
        """_summary_

        Args:
            atoms (_type_, optional): _description_. Defaults to None.
            d (float, optional): _description_. Defaults to 0.001.
        """
        Calculator.calculate(self, atoms)
        atoms = self.atoms
        solver = self.set_solver(atoms)
        solver.run(5, tqdm=True)