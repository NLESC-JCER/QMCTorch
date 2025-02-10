from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
import torch
from torch import optim
from types import SimpleNamespace

from ..utils import set_torch_double_precision
from ..scf.molecule import Molecule as SCF
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
        self.has_forces = False

        # default options for the SCF
        self.molecule = None
        self.scf_options = SimpleNamespace(calculator='pyscf',
                                           basis='dzp',
                                           scf='hf')
        
        # default options for the WF
        self.wf = None
        self.wf_options = SimpleNamespace(kinetic='jacobi',
                                          configs='single_double(2,2)',
                                          orthogonalize_mo=True,
                                          include_all_mo=True,
                                          cuda=self.use_cuda,
                                          jastrow=SimpleNamespace(
                                              kernel=PadeJastrowKernel,
                                              kernel_kwargs={'w':1.00},
                                          ),
                                          backflow=None)


        # default option for the sampler
        self.sampler = None
        self.sampler_options = SimpleNamespace(nwalkers=4000, nstep=2000, 
                                               ntherm=-1, ndecor=1, step_size=0.05)
        
        self.optimizer = None

        # default option for the solver
        self.solver = None
        self.solver_options = SimpleNamespace(track=['local_energy', 'parameters'], freeze=[],
                                              loss='energy', grad='manual',
                                              ortho_mo=False, clip_loss=False,
                                              resampling={'mode': 'update','resample_every':1, 
                                                          'nstep_update':50, 'ntherm_update':-1},
                                              niter=100, tqdm=False)

    def run_scf(self):
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
        self.molecule = SCF(atom=filename, 
                                 unit='angs', 
                                 scf=self.scf_options.scf,
                                 calculator=self.scf_options.calculator, 
                                 basis=self.scf_options.basis, redo_scf=True)

    def set_wf(self):
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
        
        if self.wf_options.jastrow is not None:
            jastrow = JastrowFactor(self.molecule, self.wf_options.jastrow.kernel, 
                                    self.wf_options.jastrow.kernel_kwargs, cuda=self.use_cuda)
        else:
            jastrow = None

        if self.wf_options.backflow is not None:
            raise ValueError("Backflow is not supported yet via the ASE calculator")
        else:
            backflow = None

        self.wf = SlaterJastrow(mol=self.molecule,
                                kinetic=self.wf_options.kinetic,
                                configs=self.wf_options.configs,
                                backflow=backflow,
                                jastrow=jastrow,
                                orthogonalize_mo=self.wf_options.orthogonalize_mo,
                                include_all_mo=self.wf_options.include_all_mo,
                                cuda=self.use_cuda)
        
    def set_sampler(self):
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
        
        self.sampler = Metropolis(nwalkers=self.sampler_options.nwalkers, nstep=self.sampler_options.nstep, 
                                  nelec=self.wf.nelec, ntherm=self.sampler_options.ntherm, ndecor=self.sampler_options.ndecor,
                                  step_size=self.sampler_options.step_size, init=self.molecule.domain('atomic'), cuda=self.use_cuda)
        
    def set_default_optimizer(self):
        if self.wf is None:
            raise ValueError("Wave function object is not set")
        lr_dict = [{'params': self.wf.jastrow.parameters(), 'lr': 1E-2},
                {'params': self.wf.ao.parameters(), 'lr': 1E-2},
                {'params': self.wf.mo.parameters(), 'lr': 1E-2},
                {'params': self.wf.fc.parameters(), 'lr': 1E-2}]
        self.optimizer = optim.Adam(lr_dict, lr=1E-2)

    def set_solver(self):
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
        if self.molecule is None:
            self.run_scf()

        if self.wf is None:
            self.set_wf()
        
        if self.sampler is None:
            self.set_sampler()

        if self.optimizer is None:
            self.set_default_optimizer()
        

        self.solver = Solver(wf=self.wf, sampler=self.sampler, optimizer=self.optimizer, scheduler=None)
        self.solver.configure(track=self.solver_options.track, freeze=self.solver_options.freeze,
                loss=self.solver_options.loss, grad=self.solver_options.grad,
                ortho_mo=self.solver_options.ortho_mo, clip_loss=self.solver_options.clip_loss,
                resampling=self.solver_options.resampling
                )

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
        self.wf = None
        self.molecule = None
        self.sampler = None
        self.solver = None
        self.has_forces = False
        self.reset_results()

    def reset_results(self):
        self.results = {}

    def reset_solver(self, atoms=None):
        """
        Update the calculator.

        This method checks if the solver has been set. If not, it sets the atoms object
        (if provided) and initializes the solver. If the solver has been set, it checks
        if the atomic positions have changed. If they have, it resets the calculator and
        sets the new atoms object and the solver again.

        Parameters
        ----------
        atoms : ASE Atoms object, optional
            The atoms object to be set. If not provided, the calculator will not be reset.

        Notes
        -----
        This method is typically called before calculating a quantity.
        """
        # if we don't have defined a solver yet
        if self.solver is None:
            if atoms is not None:
                self.set_atoms(atoms)
            self.set_solver()

        # if we do have a solver in place
        else:
            if atoms is not None:
                if (self.atoms.get_positions() !=  atoms.get_positions()).any():
                    self.reset()
                    self.set_atoms(atoms)
                    self.set_solver()

        self.reset()
        self.set_atoms(atoms)
        self.set_solver()

    def calculate(self, atoms=None, properties=['energy']):
        """_summary_

        Args:
            atoms (_type_, optional): _description_. Defaults to None.
            properties (list, optional): _description_. Defaults to ['energy'].
            system_changes (_type_, optional): _description_. Defaults to None.
        """
        # reset the solver if needed
        self.reset_solver(atoms=atoms)

        # check properties that are needed
        if any([p not in self.implemented_properties for p in properties]):
            raise ValueError('property not recognized')
        
        # compute
        for p in properties:
            if p == 'forces':
                self.calculate_forces(atoms=atoms)

            elif p == 'energy':
                self.calculate_energy(atoms=atoms)

    def calculate_energy(self, atoms=None):
        """_summary_

        Args:
            atoms (_type_, optional): _description_. Defaults to None.
        """
        # check if reset is necessary 
        self.reset_solver(atoms=atoms)

        # set wf param for opt
        self.solver.set_params_requires_grad(wf_params=True, geo_params=False)

        # run the opt
        self.solver.run(self.solver_options.niter, tqdm=self.solver_options.tqdm)

        # compute the energy 
        observable = self.solver.single_point()

        # store and output
        self.results['energy'] = observable.energy
        return self.results['energy']

    def calculate_forces(self, atoms=None):
        """_summary_

        Args:
            atoms (_type_, optional): _description_. Defaults to None.
            d (float, optional): _description_. Defaults to 0.001.
        """
        print('+++++++++++++++++++COMPUTE FORCE++++++++++++++++++++++++++++')
        # check if reset is necessary
        self.reset_solver(atoms=atoms)

        # optimize the wave function
        self.solver.set_params_requires_grad(wf_params=True, geo_params=False)
        self.solver.run(self.solver_options.niter, tqdm=self.solver_options.tqdm)

        # resample
        observable = self.solver.single_point()

        # compute the forces
        self.solver.set_params_requires_grad(wf_params=False, geo_params=True)
        _, _ = self.solver.evaluate_gradient(observable.pos)

        # store and output
        self.results['energy'] = observable.energy.cpu().numpy()
        self.results['forces'] = self.solver.wf.ao.atom_coords.grad.cpu().numpy()
        self.has_forces = True
        return self.results['forces']

    def check_forces(self):
        """
        Check if the forces have been computed.

        Returns
        -------
        bool
            True if the forces have been computed, False otherwise.
        """
        if (self.has_forces) and ('forces' in self.results):
            return True
        self.has_forces = False
        return False
    
    def get_forces(self, atoms=None):
        """
        Return the total forces.

        Parameters
        ----------
        atoms : ase.Atoms
            The ASE atoms object. If not provided, the internal atoms object is used.

        Returns
        -------
        forces : array
            The total forces on the atoms.
        """
        print(atoms.get_positions())
        print(self.atoms.get_positions())
        # if self.check_forces():
        #     return self.results['forces']
        # else:
        return self.calculate_forces(atoms=atoms)
        
    def get_total_energy(self, atoms=None):
        """
        Return the total energy.

        Parameters
        ----------
        atoms : ASE Atoms object, optional
            The atoms object to be used for the calculation.

        Returns
        -------
        energy : float
            The total energy of the system.
        """
        if 'energy' in self.results:
            return self.results['energy']
        else:
            return self.calculate_energy(atoms=atoms)