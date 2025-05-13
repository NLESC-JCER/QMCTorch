from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
import numpy as np
import torch
from torch import optim
from types import SimpleNamespace

from ..utils import set_torch_double_precision
from ..utils.constants import ANGS2BOHR
from ..scf.molecule import Molecule as SCF
from ..wavefunction.slater_jastrow import SlaterJastrow
from ..wavefunction.jastrows.elec_elec import JastrowFactor, PadeJastrowKernel
from ..wavefunction.orbitals.backflow import (
    BackFlowTransformation,
    BackFlowKernelInverse,
)
from ..solver import Solver
from ..sampler import Metropolis
from ..sampler.symmetry import C1
from .. import log


class QMCTorch(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        restart: str = None,
        *,
        labels: list = None,
        atoms: Atoms = None,
        **kwargs: dict
    ) -> None:
        """
        Initialize a QMCTorchCalculator object.

        Parameters
        ----------
        restart : str, optional
            Filename to read the calculator from.  If not given,
            an initial calculation will be performed.
        labels : list of str, optional
            List of labels.  If not given, atoms will be used
            to set initial labels.
        atoms : Atoms object, optional
            The initial atomic configuration.
        **kwargs : dict
            Additional keyword arguments are passed to the
            SCF, WF, Sampler, Optimizer and Solver objects.

        Returns
        -------
        None
        """
        Calculator.__init__(self, restart=restart, labels=labels, atoms=atoms)
        self.use_cuda = torch.cuda.is_available()
        set_torch_double_precision()
        self.has_forces = False

        # default options for the SCF
        self.molecule = None
        self.scf_options = SimpleNamespace(calculator="pyscf", basis="dzp", scf="hf")
        self.recognized_scf_options = list(self.scf_options.__dict__.keys())

        # default options for the WF
        self.wf = None
        self.wf_options = SimpleNamespace(
            kinetic="jacobi",
            configs="single_double(2,2)",
            orthogonalize_mo=True,
            mix_mo=False,
            include_all_mo=True,
            cuda=self.use_cuda,
            jastrow=SimpleNamespace(
                kernel=PadeJastrowKernel,
                kernel_kwargs={"w": 1.00},
            ),
            backflow=SimpleNamespace(
                kernel=BackFlowKernelInverse,
                kernel_kwargs={"weight": 1.00},
            ),
            gto2sto=False,
        )

        self.recognized_wf_options = list(self.wf_options.__dict__.keys())
        self.recognized_jastrow_options = list(self.wf_options.jastrow.__dict__.keys())
        self.recognized_backflow_options = list(
            self.wf_options.backflow.__dict__.keys()
        )
        self.wf_options.backflow = None

        # default option for the sampler
        self.sampler = None
        self.sampler_options = SimpleNamespace(
            nwalkers=4000,
            nstep=2000,
            ntherm=-1,
            ndecor=1,
            step_size=0.05,
            symmetry=None,
        )
        self.recognized_sampler_options = list(self.sampler_options.__dict__.keys())

        # optimizer ....
        self.optimizer = None

        # default option for the solver
        self.solver = None
        self.solver_options = SimpleNamespace(
            track=["local_energy", "parameters"],
            freeze=[],
            loss="energy",
            grad="manual",
            ortho_mo=False,
            clip_loss=False,
            resampling=SimpleNamespace(
                mode="update", resample_every=1, nstep_update=50, ntherm_update=-1
            ),
            niter=100,
            tqdm=False,
        )
        self.recognized_solver_options = list(self.solver_options.__dict__.keys())
        self.recognized_resampling_options = list(
            self.solver_options.resampling.__dict__.keys()
        )

        # default symmetry
        self.symmetry = C1()

    @staticmethod
    def validate_options(
        options: SimpleNamespace, recognized_options: list, name: str = ""
    ) -> None:
        """
        Validate that the options provided are valid.

        Parameters
        ----------
        options : SimpleNamespace
            The options to be validated.
        recognized_options : list
            The recognized options.
        name : str, optional
            The name of the options to be validated.

        Raises
        ------
        ValueError
            If the options contain invalid options.

        Returns
        -------
        None
        """
        for opt in list(options.__dict__.keys()):
            if opt not in recognized_options:
                raise ValueError(
                    "Invalid %s options: %s. Recognized options are %s"
                    % (name, opt, recognized_options)
                )

    def run_scf(self) -> None:
        """
        Set a default molecule called SCF here object. If the atoms object is not set, it raises
        a ValueError.

        The default molecule is created by writing the atoms object to a file
        named '<name>.xyz' and then loading this file into a Molecule(SCF)
        object.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.validate_options(self.scf_options, self.recognized_scf_options, "SCF")

        if self.atoms is None:
            raise ValueError("Atoms object is not set")
        filename = self.atoms.get_chemical_formula() + ".xyz"
        self.atoms.write(filename)
        self.molecule = SCF(
            atom=filename,
            unit="angs",
            scf=self.scf_options.scf,
            calculator=self.scf_options.calculator,
            basis=self.scf_options.basis,
            redo_scf=True,
        )

    def set_wf(self) -> None:
        """
        Set the default wave function for the QMCTorchCalculator.

        This method initializes a Slater-Jastrow wave function for the current molecule.
        It uses a specific configuration for the wave function and sets up a Jastrow
        factor with a PadeJastrowKernel. The method requires that a molecule object
        is already set; otherwise, it raises a ValueError.

        Raises:
            ValueError: If the molecule object is not set.
        """
        # check if molecuyle is set
        if self.molecule is None:
            raise ValueError("Molecule object is not set")

        # check jastrow and set it
        if self.wf_options.jastrow is not None:
            self.validate_options(
                self.wf_options.jastrow, self.recognized_jastrow_options, "Jastrow"
            )
            jastrow = JastrowFactor(
                self.molecule,
                self.wf_options.jastrow.kernel,
                self.wf_options.jastrow.kernel_kwargs,
                cuda=self.use_cuda,
            )
        else:
            jastrow = None

        # check backflow and set it
        if self.wf_options.backflow is not None:
            self.validate_options(
                self.wf_options.backflow, self.recognized_backflow_options, "Backflow"
            )
            backflow = BackFlowTransformation(
                self.molecule,
                self.wf_options.backflow.kernel,
                self.wf_options.backflow.kernel_kwargs,
                cuda=self.use_cuda,
            )
        else:
            backflow = None

        # checlk wf options and set wf
        self.validate_options(self.wf_options, self.recognized_wf_options, "WF")
        self.wf = SlaterJastrow(
            mol=self.molecule,
            kinetic=self.wf_options.kinetic,
            configs=self.wf_options.configs,
            backflow=backflow,
            jastrow=jastrow,
            mix_mo=self.wf_options.mix_mo,
            orthogonalize_mo=self.wf_options.orthogonalize_mo,
            include_all_mo=self.wf_options.include_all_mo,
            cuda=self.use_cuda,
        )

        # in case we want a sto transform
        if self.wf_options.gto2sto:
            if self.scf_options.calculator != "pyscf":
                raise ValueError("gto2sto is only supported for pyscf")
            self.wf = self.wf.gto2sto()

    def set_sampler(self) -> None:
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
        self.validate_options(
            self.sampler_options, self.recognized_sampler_options, "Sampler"
        )
        self.sampler = Metropolis(
            nwalkers=self.sampler_options.nwalkers,
            nstep=self.sampler_options.nstep,
            nelec=self.wf.nelec,
            ntherm=self.sampler_options.ntherm,
            ndecor=self.sampler_options.ndecor,
            step_size=self.sampler_options.step_size,
            init=self.molecule.domain("atomic"),
            symmetry=self.sampler_options.symmetry,
            cuda=self.use_cuda,
        )

    def set_default_optimizer(self) -> None:
        if self.wf is None:
            raise ValueError("Wave function object is not set")
        lr_dict = [
            {"params": self.wf.jastrow.parameters(), "lr": 1e-2},
            {"params": self.wf.ao.parameters(), "lr": 1e-2},
            {"params": self.wf.mo.parameters(), "lr": 1e-2},
            {"params": self.wf.fc.parameters(), "lr": 1e-2},
        ]
        self.optimizer = optim.Adam(lr_dict, lr=1e-2)

    def set_resampling_options(self) -> None:
        """
        Configure the resampling options for the solver.

        This method sets the number of Monte Carlo steps (`nstep_update`) to be used
        during the resampling process based on the current sampler and solver options.
        It calculates the number of sampling steps after thermalization and updates
        the `nstep_update` value if the resampling mode is 'update'.

        Notes
        -----
        - The method will adjust `nstep_update` only if the `ntherm` value is not -1
        and the resampling mode is set to 'update'.
        - The calculation for `nstep_update` considers the difference between `nstep`
        and `ntherm`, added to `ntherm_update`.

        """

        if (self.sampler_options.ntherm != -1) and (
            self.solver_options.resampling.mode == "update"
        ):
            nsample = self.sampler_options.nstep - self.sampler_options.ntherm
            self.solver_options.resampling.nstep_update = (
                self.solver_options.resampling.ntherm_update + nsample
            )

        elif (self.sampler_options.ntherm == -1) and (
            self.solver_options.resampling.mode == "update"
        ):
            if self.solver_options.resampling.ntherm_update != -1:
                self.solver_options.resampling.ntherm_update = -1

    def initialize(self) -> None:
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

        self.validate_options(
            self.solver_options, self.recognized_solver_options, "Solver"
        )
        self.validate_options(
            self.solver_options.resampling,
            self.recognized_resampling_options,
            "Resampling",
        )

        self.solver = Solver(
            wf=self.wf, sampler=self.sampler, optimizer=self.optimizer, scheduler=None
        )
        self.set_resampling_options()

        self.solver.configure(
            track=self.solver_options.track,
            freeze=self.solver_options.freeze,
            loss=self.solver_options.loss,
            grad=self.solver_options.grad,
            ortho_mo=self.solver_options.ortho_mo,
            clip_loss=self.solver_options.clip_loss,
            resampling=self.solver_options.resampling.__dict__,
        )

    def set_atoms(self, atoms: Atoms) -> None:
        """
        Set atoms object.

        Parameters
        ----------
        atoms : ASE Atoms object
            The atoms object to be set.
        """
        self.atoms = atoms

    def reset(self) -> None:
        """
        Reset the internal state of the QMCTorchCalculator.

        This method resets the internal state of the calculator by clearing
        the current atoms, wave function, molecule, sampler, and solver objects.
        It also sets the `has_forces` attribute to False and calls `reset_results`
        to clear the results dictionary. This is typically used to reinitialize
        the calculator to a clean state before performing new calculations.
        """
        self.atoms = None
        self.wf = None
        self.molecule = None
        self.sampler = None
        self.solver = None
        self.has_forces = False
        self.reset_results()

    def reset_results(self) -> None:
        """
        Reset the results dictionary.

        This method clears the current results stored in the calculator by
        setting the results dictionary to an empty state. It is typically
        used when reinitializing the calculator or after a calculation to
        ensure that previous results do not affect future computations.
        """
        self.results = {}

    def reset_solver(self, atoms: Atoms = None, force: bool = True) -> None:
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
        force : bool
            If True, the solver will be reset even if the atomic positions have not changed.

        Returns
        -------
        None

        Notes
        -----
        This method is typically called before calculating a quantity.
        """
        if atoms is not None:
            if not np.allclose(
                self.atoms.get_positions() * ANGS2BOHR,
                np.array(self.molecule.atom_coords),
            ):
                self.reset()
                self.set_atoms(atoms)
                self.initialize()
        else:
            if self.solver is None:
                self.initialize()

    def calculate(
        self,
        atoms: Atoms = None,
        properties: list = ["energy"],
        system_changes: any = None,
    ) -> float:
        """
        Calculate specified properties for the given atomic configuration.

        This method computes the requested properties, such as energy or forces,
        for the provided Atoms object. It ensures the solver is reset if the atomic
        configuration changes and checks that all requested properties are implemented.

        Parameters
        ----------
        atoms : ASE Atoms object, optional
            The atomic configuration for which the properties should be calculated.
            If not provided, the current atoms object associated with the calculator
            is used.
        properties : list of str, optional
            A list of properties to calculate. Supported properties are 'energy'
            and 'forces'. Default is ['energy'].
        system_changes : any, optional
            Information about the changes in the atomic system. Default is None.

        Returns
        -------
        float
            The computed value of the requested property.

        Raises
        ------
        ValueError
            If a requested property is not recognized or not implemented.

        Notes
        -----
        The method first resets the solver if needed, checks the validity of the
        requested properties, and then computes each property one-by-one.
        """

        # reset the solver if needed
        self.reset_solver(atoms=atoms)

        # check properties that are needed
        if any([p not in self.implemented_properties for p in properties]):
            raise ValueError("property not recognized")

        # compute
        for p in properties:
            if p == "forces":
                return self._calculate_forces(atoms=atoms)

            elif p == "energy":
                return self._calculate_energy(atoms=atoms)

    def _calculate_energy(self, atoms: Atoms = None) -> float:
        # check if reset is necessary
        """
        Compute the energy using the wave function and the atomic positions.

        Parameters
        ----------
        atoms : ASE Atoms object, optional
            The atoms object to be used for the computation. If not provided, the calculator
            will use the atoms object that was set when the calculator was created.

        Returns
        -------
        energy : float
            The computed energy.

        Notes
        -----
        This method first resets the solver (if necessary), then sets the wave function parameters
        as the only parameters that require gradient computation. It then runs the optimization
        for the specified number of iterations (with tqdm if specified), and finally computes the
        energy of the system. The result is stored in the calculator's results dictionary and
        returned.
        """
        # optimize the wave function
        if self.solver_options.niter > 0:
            self.solver.set_params_requires_grad(wf_params=True, geo_params=False)
            self.solver.freeze_parameters(self.solver_options.freeze)
            self.solver.run(self.solver_options.niter, tqdm=self.solver_options.tqdm)

        # compute the energy
        observable = self.solver.single_point()

        # store and output
        self.results["energy"] = observable.energy
        return self.results["energy"]

    def _calculate_forces(self, atoms: Atoms = None) -> float:
        # check if reset is necessary
        """
        Compute the forces using the wave function and the atomic positions.

        Parameters
        ----------
        atoms : ASE Atoms object, optional
            The atoms object to be used for the computation. If not provided, the calculator
            will use the atoms object that was set when the calculator was created.

        Returns
        -------
        forces : numpy.ndarray
            The computed forces, with the same shape as the positions of the atoms object.

        Notes
        -----
        The forces are computed by optimizing the wave function using the atomic positions as variational parameters.
        """

        # optimize the wave function
        if self.solver_options.niter > 0:
            self.solver.set_params_requires_grad(wf_params=True, geo_params=False)
            self.solver.freeze_parameters(self.solver_options.freeze)
            self.solver.run(self.solver_options.niter, tqdm=self.solver_options.tqdm)

        # resample
        observable = self.solver.single_point()

        # compute the forces
        forces = (
            self.solver.compute_forces(self.symmetry(observable.pos))
            .detach()
            .cpu()
            .numpy()
        )

        # store and output
        self.results["energy"] = observable.energy.cpu().numpy()
        self.results["forces"] = forces
        self.solver.wf.zero_grad()

        self.has_forces = True
        return self.results["forces"]

    def check_forces(self) -> bool:
        """
        Check if the forces have been computed.

        Returns
        -------
        bool
            True if the forces have been computed, False otherwise.
        """
        if (self.has_forces) and ("forces" in self.results):
            return True
        self.has_forces = False
        return False

    def get_forces(self, atoms: Atoms = None) -> np.ndarray:
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

        self.reset_solver(atoms=atoms)
        if self.check_forces():
            return self.results["forces"]
        else:
            return self._calculate_forces(atoms=atoms)

    def get_total_energy(self, atoms: Atoms = None) -> float:
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
        self.reset_solver(atoms=atoms)
        if "energy" in self.results:
            return self.results["energy"]
        else:
            return self._calculate_energy(atoms=atoms)
