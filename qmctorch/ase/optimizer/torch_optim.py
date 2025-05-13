from typing import IO, Optional, Union
from types import SimpleNamespace
from torch.optim import SGD
from torch.optim import Optimizer as torch_optimizer
import numpy as np
import time
from math import sqrt
from copy import deepcopy
from ase import Atoms
from ase.optimize.optimize import Optimizer
from ...utils.constants import BOHR2ANGS


class TorchOptimizer(Optimizer):
    def __init__(
        self,
        atoms: Atoms,
        optimizer: Optional[torch_optimizer] = None,
        nepoch_wf_init: Optional[int] = 100,
        nepoch_wf_update: Optional[int] = 10,
        batchsize: Optional[int] = None,
        tqdm: Optional[bool] = False,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = "-",
        trajectory: Optional[str] = None,
        master: Optional[bool] = None,
    ):
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

        self.opt_geo = optimizer
        self.batchsize = batchsize
        self.tqdm = tqdm
        self.nepoch_wf_init = nepoch_wf_init
        self.nepoch_wf_update = nepoch_wf_update
        self.xyz_trajectory = None

    def log(self, e: float, forces: np.ndarray) -> float:
        """
        Write to the log file.

        Parameters
        ----------
        e : float
            Energy of the system.
        forces : np.ndarray
            Forces on the atoms.

        Returns
        -------
        fmax : float
            Maximum force on any atom.

        Notes
        -----
        This function is called by the optimizer at each step. It writes the
        energy, forces, and time to the log file.
        """
        fmax = sqrt((forces**2).sum(axis=1).max())
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (" " * len(name), "Step", "Time", "Energy", "fmax")
                msg = "%s  %4s %8s %15s  %12s\n" % args
                self.logfile.write(msg)

            args = (name, self.nsteps, T[3], T[4], T[5], e, fmax)
            msg = "%s:  %3d %02d:%02d:%02d %15.6f %15.6f\n" % args
            self.logfile.write(msg)
            self.logfile.flush()
        return fmax

    def run(
        self, fmax: float, steps: int = 10, hdf5_group: str = "geo_opt"
    ) -> SimpleNamespace:
        """
        Run a geometry optimization.

        Parameters
        ----------
        fmax : float
            Convergence criteria for the forces.
        steps : int, optional
            Number of optimization steps. Defaults to 10.
        hdf5_group : str, optional
            HDF5 group where the data is stored. Defaults to 'geo_opt'.

        Returns
        -------
        observable : Observable
            The observable instance containing the optimized geometry.

        Notes
        -----
        The geometry optimization is done in two steps. First, the wave function
        is optimized using the method specified in the Solver instance, then
        the geometry is optimized using the provided optimizer. The process is
        repeated until convergence or until the specified number of iterations
        is reached.
        """
        solver = self.atoms.calc.solver

        if self.opt_geo is None:
            self.opt_geo = SGD(solver.wf.parameters(), lr=1e-2)
            self.opt_geo.lpos_needed = False

        # save the optimizer used for the wf params
        self.opt_wf = deepcopy(solver.opt)
        self.opt_wf.lpos_needed = solver.opt.lpos_needed

        # save the grad method
        self.eval_grad_wf = solver.evaluate_gradient

        # log data
        solver.prepare_optimization(self.batchsize, None, self.tqdm)
        solver.log_data_opt(steps, "geometry optimization")

        # init the traj
        self.xyz_trajectory = [solver.wf.geometry(None)]

        # initial wf optimization
        solver.set_params_requires_grad(wf_params=True, geo_params=False)
        solver.freeze_parameters(solver.freeze_params_list)
        solver.run_epochs(self.nepoch_wf_init)

        for n in range(steps):
            # one step of geo optim
            solver.set_params_requires_grad(wf_params=False, geo_params=True)
            solver.opt = self.opt_geo
            solver.evaluate_gradient = (
                solver.evaluate_grad_auto
            )  # evaluate_grad_manual not valid for forces
            solver.run_epochs(1, verbose=False)
            forces = solver.wf.forces()
            print(solver.wf.geometry(None, convert_to_angs=True))
            self.xyz_trajectory.append(solver.wf.geometry(None, convert_to_angs=True))

            # make a few wf optim
            solver.set_params_requires_grad(wf_params=True, geo_params=False)
            solver.freeze_parameters(solver.freeze_params_list)
            solver.opt = self.opt_wf
            solver.evaluate_gradient = self.eval_grad_wf
            cumulative_loss = solver.run_epochs(
                self.nepoch_wf_update, with_tqdm=self.tqdm, verbose=False
            )

            # update the geometry
            self.optimizable.set_positions(
                solver.wf.geometry(None, convert_to_angs=True)
            )
            current_fmax = self.log(cumulative_loss, forces)
            self.call_observers()

            if current_fmax < fmax:
                break

        # restore the sampler number of step
        solver.restore_sampling_parameters()

        # dump
        solver.observable.geometry = self.xyz_trajectory
        solver.save_data(hdf5_group)

        return solver.observable
