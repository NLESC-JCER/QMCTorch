from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
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
                 **kwargs):
        
        Calculator.__init__(self, restart=restart, labels=labels, atoms=atoms)
        self.use_cuda = torch.cuda.is_available()
        set_torch_double_precision()

    def set(self, **kwargs):
        recpognized_options = ['molecule','wf','sampler','optimizer','solver']
        for k, _ in kwargs.items():
            if k.lower() not in recpognized_options:
                raise ValueError("Unknown option %s" % k)
            
            if k.lower() == 'molecule':
                self.set_molecule(kwargs[k])
            if k.lower() == 'wf':
                self.set_wf(kwargs[k])
            if k.lower() == 'sampler':
                self.set_sampler(kwargs[k])
            if k.lower() == 'solver':
                self.set_solver(kwargs[k])
            if k.lower() == 'optimizer':
                self.set_optimizer(kwargs[k])

    def set_molecule(self, molecule):
        """
        Set molecule object.

        Parameters
        ----------
        molecule : qmctorch.Molecule
            The molecule object to be set. The atoms object will be set
            accordingly.
        """
        self.molecule = molecule
        if molecule is not None:
            atom_names = ''.join(molecule.atoms)
            self.set_atoms(Atoms(atom_names, positions=molecule.atom_coords))

    def set_default_molecule(self):
        """
        Set a default molecule object. If the atoms object is not set, it raises
        a ValueError.

        The default molecule is created by writing the atoms object to a file
        named 'ase_molecule.xyz' and then loading this file into a Molecule
        object.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.atoms is None:
            raise ValueError("Atoms object is not set")
        filename = 'ase_molecule.xyz'
        self.atoms.write(filename) 
        self.molecule = Molecule(atom=filename, unit='angs', calculator='pyscf', basis='dzp')

    def set_default_wf(self):
        """
        Set the default wave function for the QMCTorchCalculator.

        This method initializes a Slater-Jastrow wave function for the current molecule.
        It uses a specific configuration for the wave function and sets up a Jastrow
        factor with a PadeJastrowKernel. The method requires that a molecule object
        is already set; otherwise, it raises a ValueError.

        Raises:
            ValueError: If the molecule object is not set.
        """

        if self.molecule is None:
            raise ValueError("Molecule object is not set")
        
        configs =  'single_double(2,2)'
        jastrow = JastrowFactor(self.molecule, PadeJastrowKernel, kernel_kwargs={'w':1.00}, cuda=self.use_gpu)
        self.wf = SlaterJastrow(mol=self.molecule,
                                kinetic='jacobi',
                                configs=configs,
                                backflow=None,
                                jastrow=jastrow,
                                orthogonalize_mo=True,
                                cuda=self.use_gpu)
        
    def set_default_sampler(self):
        """
        Set default sampler object.

        Parameters
        ----------
        None

        Notes
        -----
        The default sampler object is a Metropolis object with 4000 walkers,
        2000 steps, a step size of 0.05, and one decorrelation step.
        The sampler is initialized with atomic positions.
        If self.use_gpu is True, the sampler will use the GPU.
        """
        if self.wf is None:
            raise ValueError("Wave function object is not set")
        
        self.sampler = Metropolis(nwalkers=4000, nstep=2000, nelec=self.wf.nelec, ntherm=-1, ndecor=1,
                    step_size=0.05, init=self.mol.domain('atomic'), cuda=self.use_gpu)
        
    def set_default_optimizer(self):
        if self.wf is None:
            raise ValueError("Wave function object is not set")
        lr_dict = [{'params': self.wf.jastrow.parameters(), 'lr': 1E-2},
                {'params': self.wf.ao.parameters(), 'lr': 1E-2},
                {'params': self.wf.mo.parameters(), 'lr': 1E-2},
                {'params': self.wf.fc.parameters(), 'lr': 1E-2}]
        self.optimizer = optim.Adam(lr_dict, lr=1E-2)

    def set_default_solver(self):
        if self.wf is None:
            self.set_default_wf()
        
        if self.sampler is None:
            self.set_default_sampler()

        if self.optimizer is None:
            self.set_default_optimizer()
        

        self.solver = Solver(wf=self.wf, sampler=self.sampler, optimizer=self.optimizer, scheduler=None)
        self.solver.set_params_requires_grad(wf_params=True, geo_params=False)

        self.solver.configure(track=['local_energy', 'parameters'], freeze=[],
                loss='energy', grad='manual',
                ortho_mo=False, clip_loss=False,
                resampling={'mode': 'update','resample_every':1, 'nstep_update':50, 'ntherm_update':-1}
                )


    def set_wf(self, wf):
        """
        Set the wave function object.

        Parameters
        ----------
        wf : qmctorch.WaveFunction
            The wave function object to be set.
        """
        self.wf = wf
        self.set_molecule(self.wf.molecule)

    def set_sampler(self, sampler):
        """
        Set the sampler object.

        Parameters
        ----------
        sampler : qmctorch.Sampler
            The sampler object to be set.
        """
        self.sampler = sampler

    def set_optimizer(self, optimizer):
        """
        Set optimizer object.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer object to be set.
        """
        self.optimizer = optimizer

    def set_solver(self, solver):
        """
        Set the solver object.

        Parameters
        ----------
        solver : qmctorch.Solver
            The solver object to be set.
        """
        self.solver = solver
        self.set_wf(self.solver.wf)
        self.set_sampler(self.solver.sampler)
        self.set_optimizer(self.solver.optimizer)
        

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
            if p == 'energy':
                self.calculate_energy(atoms=atoms)
            if p == 'froces':
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