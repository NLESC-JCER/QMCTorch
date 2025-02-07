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

    def update_molecule(self, atoms):
        """
        Update the molecule object based on the current atoms object.

        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to be used to update the molecule object.

        Returns
        -------
        None
        """
        if self.molecule is None:
            raise ValueError('Molecule object not set')
        
        self.atoms = atoms
        filename = 'ase_molecule.xyz'
        self.atoms.write(filename) 
        self.molecule = Molecule(atom=filename, unit=self.molecule.unit, 
                                 calculator=self.molecule.calculator, basis=self.molecule.basis_name,
                                 scf=self.molecule.scf_level, charge=self.molecule.charge, spin=self.molecule.spin,
                                 name=self.molecule.name)

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
        jastrow = JastrowFactor(self.molecule, PadeJastrowKernel, kernel_kwargs={'w':1.00}, cuda=self.use_cuda)
        self.wf = SlaterJastrow(mol=self.molecule,
                                kinetic='jacobi',
                                configs=configs,
                                backflow=None,
                                jastrow=jastrow,
                                orthogonalize_mo=True,
                                cuda=self.use_cuda)
        

    def update_wf(self):
        """
        Updates the wave function using the current molecule and the
        previously set wave function configuration parameters.

        Raises:
            ValueError: If the wave function object is not set yet.
        """
        if self.wf is None:
            raise ValueError("Wave function object not set yet")
        
        self.wf(self.molecule, configs=self.wf.configs_method, 
                kinetic=self.wf.kinetic_method,
                backflow=None, jastrow=self.wf.jastrow,
                orthogonalize_mo=self.wf.orthogonalize_mo,
                cuda=self.use_cuda
                )
        
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
        If self.use_cud is True, the sampler will use the GPU.
        """
        if self.wf is None:
            raise ValueError("Wave function object is not set")
        
        self.sampler = Metropolis(nwalkers=4000, nstep=2000, nelec=self.wf.nelec, ntherm=-1, ndecor=1,
                    step_size=0.05, init=self.mol.domain('atomic'), cuda=self.use_cuda)
        
    def set_default_optimizer(self):
        if self.wf is None:
            raise ValueError("Wave function object is not set")
        lr_dict = [{'params': self.wf.jastrow.parameters(), 'lr': 1E-2},
                {'params': self.wf.ao.parameters(), 'lr': 1E-2},
                {'params': self.wf.mo.parameters(), 'lr': 1E-2},
                {'params': self.wf.fc.parameters(), 'lr': 1E-2}]
        self.optimizer = optim.Adam(lr_dict, lr=1E-2)

    def set_default_solver(self):
        """
        Set the default solver object for the QMCTorchCalculator.

        This method initializes the default Solver object for the QMCTorchCalculator.
        It first checks if the wave function, sampler, and optimizer objects are set,
        and if not, it initializes them with default values. It then sets up the Solver
        object with those defaults. The method also sets the parameters that require
        gradient computation and the configuration for the solver.

        Parameters
        ----------
        None

        Notes
        -----
        The default configuration for the solver is set to track the local energy and
        the parameters of the wave function, with no frozen parameters. The gradient
        computation is set to manual, and the resampling is set to update every step.
        """
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
        """_summary_

        Args:
            atoms (_type_): _description_
        """
        self.solver = solver
         

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



    def calculate_energy(self, atoms=None):
        """_summary_

        Args:
            atoms (_type_, optional): _description_. Defaults to None.
        """
        if atoms is not None:
            self.atoms = atoms
            if self.molecule is None:
                self.set_default_molecule()
            else:
                self.update_molecule()

            if self.wf is None:
                self.set_default_wf()
            else:
                self.update_wf()
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