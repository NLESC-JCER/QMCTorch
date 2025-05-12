import os
import numpy as np
from typing import Dict, List
from mendeleev import element
from types import SimpleNamespace
import h5py

from .calculator import CalculatorADF, CalculatorADF2019
from .calculator import CalculatorPySCF

from ..utils import dump_to_hdf5, load_from_hdf5, bytes2str
from ..utils.constants import ANGS2BOHR
from .. import log

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    log.info("  MPI not found.")


class Molecule:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        atom: str = None,
        calculator: str = "adf",
        scf: str = "hf",
        basis: str = "dzp",
        unit: str = "bohr",
        charge: int = 0,
        spin: int = 0,
        name: str = None,
        load: str = None,
        save_scf_file: bool = False,
        redo_scf: bool = False,
        rank: int = 0,
        mpi_size: int = 0,
    ) -> None:
        """Create a molecule in QMCTorch

        Args:
            atom (str, optional): Defines the atoms and their positions. Defaults to None.
                - At1 x y z; At2 x y z ... : Provide the atomic coordinate directly
                - <file>.xyz : Provide the path to an .xyz file containing the atomic coordinates
            calculator (str, optional): Select SCF calculator. Defaults to 'adf'.
                - 'pyscf' : PySCF calculator
                - 'adf' : ADF2020+ calculator
                - 'adf2019' : ADF2019 calculator
            scf (str, optional): Select SCF level of theory. Defaults to 'hf'.
                - 'hf' : Hartree-Fock calculation
                - 'dft' : Density Functional Theory using LDA
            charge (int, optional): Extra charge on the molecule. Defaults to 0.
            spin (int, optional): Excess of spin-up electrons on the molecule. Defaults to 0.
            basis (str, optional): Select the basis set. Defaults to 'dzp'.
            unit (str, optional): Units of the coordinates; 'bohr' or 'angs'. Defaults to 'bohr'.
            name (str, optional): Name of the molecule. Defaults to None.
            load (str, optional): Path to a HDF5 file to load. Defaults to None.
            save_scf_file (bool, optional): Save the SCF file (when applicable). Defaults to False.
            redo_scf (bool, optional): If true, ignore existing HDF5 file and redo SCF calculation.
            rank (int, optional): Rank of the process. Defaults to 0.
            mpi_size (int, optional): Size of the MPI world.

        Returns:
            None

        Examples:
            >>> from qmctorch.scf import Molecule
            >>> mol = Molecule(atom='H 0 0 0; H 0 0 1', unit='angs',
            ...                calculator='adf', basis='dzp')
        """

        self.atom_coords: list = []
        self.atomic_nelec: list = []
        self.atomic_number: list = []
        self.atoms: list = []
        self.atoms_str: str = atom
        self.hdf5file: str = None
        self.max_angular: int = 2
        self.name: str = name
        self.natom: int = 0
        self.ndown: int = 0
        self.nelec: int = 0
        self.nup: int = 0
        self.charge: int = charge
        self.spin: int = spin
        self.unit: str = unit
        self.basis: SimpleNamespace = SimpleNamespace()
        self.calculator_name: str = calculator
        self.basis_name: str = basis
        self.save_scf_file: bool = save_scf_file
        self.scf_level: str = scf

        if rank == 0:
            log.info("")
            log.info(" SCF Calculation")

        # load an existing hdf5 file
        if load is not None:
            log.info("  Loading data from {file}", file=load)
            self._load_hdf5(load)
            self.hdf5file = load

        else:
            # extract the atom names/positions from
            # the atom kwargs
            self._process_atom_str()

            # name of the hdf5 file
            self.hdf5file = "_".join([self.name, calculator, basis]) + ".hdf5"

            if rank == 0:
                if self.unit not in ["angs", "bohr"]:
                    raise ValueError("unit should be angs or bohr")

                # force a redo of the sc calculation
                if os.path.isfile(self.hdf5file) and redo_scf:
                    log.info(
                        "  Removing {file} and redo SCF calculations",
                        file=self.hdf5file,
                    )
                    os.remove(self.hdf5file)

                # deals with existing files
                if os.path.isfile(self.hdf5file):
                    log.info("  Reusing scf results from {file}", file=self.hdf5file)
                    self.basis = self._load_basis()

                # perform the scf calculation
                else:
                    log.info("  Running scf  calculation")

                    calc = {
                        "adf2019": CalculatorADF2019,
                        "adf": CalculatorADF,
                        "pyscf": CalculatorPySCF,
                    }[calculator]

                    self.calculator = calc(
                        self.atoms,
                        self.atom_coords,
                        basis,
                        self.charge,
                        self.spin,
                        self.scf_level,
                        self.unit,
                        self.name,
                        self.save_scf_file,
                    )

                    self.basis = self.calculator.run()
                    self.save_scf_file = self.calculator.savefile

                    dump_to_hdf5(self, self.hdf5file, root_name="molecule")

                    self._check_basis()
                    self.log_data()

            if mpi_size != 0:
                MPI.COMM_WORLD.barrier()

                if rank != 0:
                    log.info("  Loading data from {file}", file=self.hdf5file)
                    self._load_hdf5(self.hdf5file)

    def log_data(self) -> None:
        log.info("  Molecule name       : {0}", self.name)
        log.info("  Number of electrons : {0}", self.nelec)
        log.info("  SCF calculator      : {0}", self.calculator_name)
        log.info("  Basis set           : {0}", self.basis_name)
        log.info("  SCF                 : {0}", self.scf_level.upper())
        log.info("  Number of AOs       : {0}", self.basis.nao)
        log.info("  Number of MOs       : {0}", self.basis.nmo)
        log.info(
            "  SCF Energy          : {:.3f} Hartree".format(self.get_total_energy())
        )

    def domain(self, method: str) -> Dict:
        """Returns information to initialize the walkers

        Args:
            method (str): 'center', all electron at the center of the system
                          'uniform', all electrons in a cube surrounding the molecule
                          'normal',  all electrons in a sphere surrounding the molecule
                          'atomic', electrons around the atoms

        Returns:
            dict: dictionary containing corresponding information

        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> domain = mol.domain('atomic')
        """
        domain = dict()
        domain["method"] = method

        if method == "center":
            domain["center"] = np.mean(self.atom_coords, 0)

        elif method == "uniform":
            domain["min"] = np.min(self.atom_coords) - 0.5
            domain["max"] = np.max(self.atom_coords) + 0.5

        elif method == "normal":
            domain["mean"] = np.mean(self.atom_coords, 0)
            domain["sigma"] = np.diag(np.std(self.atom_coords, 0) + 0.25)

        elif method == "atomic":
            domain["atom_coords"] = self.atom_coords
            domain["atom_num"] = self.atomic_number
            domain["atom_nelec"] = self.atomic_nelec

        else:
            raise ValueError("Method to initialize the walkers not recognized")

        return domain

    def _process_atom_str(self) -> None:
        """Process the atom description."""

        if self.atoms_str.endswith(".xyz"):
            if os.path.isfile(self.atoms_str):
                atoms = self._read_xyz_file()
            else:
                raise FileNotFoundError("File %s not  found" % self.atoms_str)
        else:
            atoms = self.atoms_str.split(";")

        self._get_atomic_properties(atoms)

    def _get_atomic_properties(self, atoms: str) -> None:
        """Generates the atomic propeties of the molecule

        Args:
            atoms (str): atoms given in input
        """

        # loop over all atoms
        for a in atoms:
            atom_data = a.split()
            self.atoms.append(atom_data[0])
            x, y, z = float(atom_data[1]), float(atom_data[2]), float(atom_data[3])

            conv2bohr = 1
            if self.unit == "angs":
                conv2bohr = ANGS2BOHR
            self.atom_coords.append([x * conv2bohr, y * conv2bohr, z * conv2bohr])

            self.atomic_number.append(element(atom_data[0]).atomic_number)
            self.atomic_nelec.append(element(atom_data[0]).electrons)
            self.nelec += element(atom_data[0]).electrons

        # add extra charge on the molecule
        self.nelec += self.charge

        # size of the system
        self.natom = len(self.atoms)
        if (self.nelec - self.spin) % 2 != 0:
            raise ValueError(
                "%d electrons and spin %d doesn't make sense" % (self.nelec, self.spin)
            )
        self.nup = int((self.nelec - self.spin) / 2) + self.spin
        self.ndown = int((self.nelec - self.spin) / 2)

        # name of the system
        if self.name is None:
            self.name = self._get_mol_name(self.atoms)
        self.atoms = np.array(self.atoms)

    def _read_xyz_file(self) -> List:
        """Process a xyz file containing the data

        Returns:
            list -- atoms and xyz position
        """
        with open(self.atoms_str, "r") as f:
            data = f.readlines()

        natom = int(data[0])
        atoms = data[2 : 2 + natom]
        self.atoms_str = ""
        for a in atoms[:-1]:
            self.atoms_str += a + "; "
        self.atoms_str += atoms[-1]
        return atoms

    @staticmethod
    def _get_mol_name(atoms: List[str]) -> str:
        """
        Generate a molecule name from the list of atoms.

        Args:
            atoms (List[str]): List of atoms in the molecule

        Returns:
            str: The molecule name
        """
        mol_name = ""
        unique_atoms = list(set(atoms))
        for ua in unique_atoms:
            mol_name += ua
            nat = atoms.count(ua)
            if nat > 1:
                mol_name += str(nat)
        return mol_name

    def _load_basis(self) -> SimpleNamespace:
        """Get the basis information needed to compute the AO values."""

        h5 = h5py.File(self.hdf5file, "r")
        basis_grp = h5["molecule"]["basis"]
        self.basis = SimpleNamespace()

        self.basis.radial_type = bytes2str(basis_grp["radial_type"][()])
        self.basis.harmonics_type = bytes2str(basis_grp["harmonics_type"][()])

        self.basis.nao = int(basis_grp["nao"][()])
        self.basis.nmo = int(basis_grp["nmo"][()])

        self.basis.nshells = basis_grp["nshells"][()]
        self.basis.nao_per_atom = basis_grp["nao_per_atom"][()]
        self.basis.index_ctr = basis_grp["index_ctr"][()]
        self.basis.nctr_per_ao = basis_grp["nctr_per_ao"][()]

        self.basis.bas_exp = basis_grp["bas_exp"][()]
        self.basis.bas_coeffs = basis_grp["bas_coeffs"][()]

        self.basis.atom_coords_internal = basis_grp["atom_coords_internal"][()]

        self.basis.TotalEnergy = basis_grp["TotalEnergy"][()]
        self.basis.mos = basis_grp["mos"][()]

        if self.basis.harmonics_type == "cart":
            self.basis.bas_kr = basis_grp["bas_kr"][()]
            self.basis.bas_kx = basis_grp["bas_kx"][()]
            self.basis.bas_ky = basis_grp["bas_ky"][()]
            self.basis.bas_kz = basis_grp["bas_kz"][()]

        elif self.basis.harmonics_type == "sph":
            self.basis.bas_n = basis_grp["bas_n"][()]
            self.basis.bas_l = basis_grp["bas_l"][()]
            self.basis.bas_m = basis_grp["bas_m"][()]

        else:
            raise ValueError(
                "Harmonics type should be cart or sph \
                but %s was found in %s"
                % (self.basis.harmonics_type, self.hdf5file)
            )

        h5.close()
        return self.basis

    def print_total_energy(self) -> None:
        """Print the SCF energy of the molecule.

        Examples::
            >>> mol = Molecule('h2.xyz', calculator='adf', basis='sz')
            >>> mol.print_total_energy()
        """
        e = self.get_total_energy()
        log.info("== SCF Energy : {e}", e=e)

    def get_total_energy(self) -> float:
        """Get the value of the total energy."""
        h5 = h5py.File(self.hdf5file, "r")
        e = h5["molecule"]["basis"]["TotalEnergy"][()]
        h5.close()
        return e

    def _check_basis(self) -> None:
        """Check if the basis contains all the necessary fields."""

        names = [
            "bas_coeffs",
            "bas_exp",
            "nshells",
            "atom_coords_internal",
            "nao",
            "nmo",
            "index_ctr",
            "mos",
            "TotalEnergy",
        ]

        if self.basis.harmonics_type == "cart":
            names += ["bas_kx", "bas_ky", "bas_kz", "bas_kr"]

        elif self.basis.harmonics_type == "sph":
            names += ["bas_n", "bas_l", "bas_m"]

        for n in names:
            if not hasattr(self.basis, n):
                raise ValueError(n, " not in the basis namespace")

    def _load_hdf5(self, filename: str) -> None:
        """Load a molecule from hdf5

        Args:
            filename (str): path to the file to be loaded
        """

        # load the data
        load_from_hdf5(self, filename, "molecule")

        # cast some of the important data type
        # should be done by the hdf5_utils in the future
        self.atoms = self.atoms.astype("U")
        self.basis.nao = int(self.basis.nao)
        self.basis.nmo = int(self.basis.nmo)

        cast_fn = {
            "nelec": int,
            "nup": int,
            "ndown": int,
            "atoms": lambda x: x.astype("U"),
            "atomic_nelec": lambda x: [int(i) for i in x],
        }

        for name, fn in cast_fn.items():
            self.__setattr__(name, fn(self.__getattribute__(name)))

        cast_fn = {"nao": int, "nmo": int}

        for name, fn in cast_fn.items():
            self.basis.__setattr__(name, fn(self.basis.__getattribute__(name)))
