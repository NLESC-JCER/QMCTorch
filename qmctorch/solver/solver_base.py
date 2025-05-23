from types import SimpleNamespace
from typing import Optional, Dict, Union, List, Tuple, Any
from ..wavefunction import WaveFunction
from ..sampler import SamplerBase
import os
import numpy as np
import torch
from tqdm import tqdm

from .. import log
from ..utils import add_group_attr, dump_to_hdf5
from ..utils import get_git_tag


class SolverBase:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        wf: Optional[WaveFunction] = None,
        sampler: Optional[SamplerBase] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        output: Optional[str] = None,
        rank: int = 0,
    ) -> None:
        """Base Class for QMC solver

        Args:
            wf (qmctorch.WaveFunction, optional): wave function. Defaults to None.
            sampler (qmctorch.sampler, optional): Sampler. Defaults to None.
            optimizer (torch.optim.Optimizer, optional): optimizer. Defaults to None.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): scheduler. Defaults to None.
            output (str, optional): hdf5 filename. Defaults to None.
            rank (int, optional): rank of he process. Defaults to 0.
        """

        self.wf = wf
        self.sampler = sampler
        self.opt = optimizer
        self.scheduler = scheduler
        self.cuda = False
        self.device: torch.device = torch.device("cpu")
        self.qmctorch_version: str = get_git_tag()

        # member defined in the child and or method
        self.dataloader = None
        self.loss = None
        self.obs_dict = None

        # if pos are needed for the optimizer (obsolete ?)
        if self.opt is not None and "lpos_needed" not in self.opt.__dict__.keys():
            self.opt.lpos_needed = False

        # distributed model
        self.save_model: str = "model.pth"

        # handles GPU availability
        if self.wf.cuda:
            self.device = torch.device("cuda")
            self.sampler.cuda = True
            self.sampler.walkers.cuda = True
        else:
            self.device = torch.device("cpu")

        self.hdf5file: str = output
        if output is None:
            basename: str = os.path.basename(self.wf.mol.hdf5file).split(".")[0]
            self.hdf5file = basename + "_QMCTorch.hdf5"

        if rank == 0:
            if os.path.isfile(self.hdf5file):
                os.remove(self.hdf5file)
            dump_to_hdf5(self, self.hdf5file)

        self.log_data()

    def configure_resampling(  # pylint: disable=too-many-arguments
        self,
        mode: str = "update",
        resample_every: int = 1,
        nstep_update: int = 25,
        ntherm_update: int = -1,
        increment: Dict = {"every": None, "factor": None},
    ):
        """Configure the resampling

        Args:
            mode (str, optional): method to resample : 'full', 'update', 'never'
                                  Defaultsr to 'update'.
            resample_every (int, optional): Number of optimization steps between resampling
                                 Defaults to 1.
            nstep_update (int, optional): Number of MC steps in update mode.
                                          Defaults to 25.
            ntherm_update (int, oprrtional): Number of MC steps to thermalize the new sampling.
                                          Defaults to -1.
            increment (dict, optional): dict containing the option to increase the sampling space
                                        every (int) : increment the sampling space every n optimization step
                                        factor (int) : increment with factor x nwalkers points

        """

        self.resampling_options = SimpleNamespace()
        valid_mode = ["never", "full", "update"]
        if mode not in valid_mode:
            raise ValueError(mode, "not a valid update method : ", valid_mode)

        self.resampling_options.mode = mode
        self.resampling_options.resample_every = resample_every
        self.resampling_options.ntherm_update = ntherm_update
        self.resampling_options.nstep_update = nstep_update
        self.resampling_options.increment = increment

    def track_observable(self, obs_name: Union[str, List[str]]):
        """define the observalbe we want to track

        Args:
            obs_name (list): list of str defining the observalbe.
                             Each str must correspond to a WaveFuncion method
        """

        # make sure it's a list
        if not isinstance(obs_name, list):
            obs_name = list(obs_name)

        # sanity check
        valid_obs_name = [
            "energy",
            "local_energy",
            "geometry",
            "parameters",
            "gradients",
        ]
        for name in obs_name:
            if name in valid_obs_name:
                continue
            elif hasattr(self.wf, name):
                continue
            else:
                log.info("   Error : Observable %s not recognized" % name)
                log.info("         : Possible observable")
                for n in valid_obs_name:
                    log.info("         :  - %s" % n)
                log.info("         :  - or any method of the wave function")
                raise ValueError("Observable not recognized")

        # reset the Namesapce
        self.observable = SimpleNamespace()
        self.observable.qmctorch_version = self.qmctorch_version

        # add the energy of the sytem
        if "energy" not in obs_name:
            obs_name += ["energy"]

        # add the geometry of the system
        if "geometry" not in obs_name:
            obs_name += ["geometry"]

        for k in obs_name:
            if k == "parameters":
                # for key, p in zip(self.wf.state_dict().keys(), self.wf.parameters()):
                for key, p in self.wf.named_parameters():
                    if p.requires_grad:
                        self.observable.__setattr__(key, [])

            elif k == "gradients":
                # for key, p in zip(self.wf.state_dict().keys(), self.wf.parameters()):
                for key, p in self.wf.named_parameters():
                    if p.requires_grad:
                        self.observable.__setattr__(key + ".grad", [])

            else:
                self.observable.__setattr__(k, [])

        self.observable.models = SimpleNamespace()

    def store_observable(
        self,
        pos: torch.tensor,
        local_energy: Optional[torch.tensor] = None,
        ibatch: Optional[int] = None,
        **kwargs
    ):
        """store observale in the dictionary

        Args:
            pos (torch.tensor): positions of th walkers
            local_energy (torch.tensor, optional): precomputed values of the local
                                           energy. Defaults to None
            ibatch (int): index of the current batch. Defaults to None
        """

        if self.wf.cuda and pos.device.type == "cpu":
            pos = pos.to(self.device)

        for obs in self.observable.__dict__.keys():
            # store the energy
            if obs == "energy":
                if local_energy is None:
                    local_energy = self.wf.local_energy(pos)

                data = local_energy.cpu().detach().numpy()

                if (ibatch is None) or (ibatch == 0):
                    self.observable.energy.append(np.mean(data).item())
                else:
                    self.observable.energy[-1] *= ibatch / (ibatch + 1)
                    self.observable.energy[-1] += np.mean(data).item() / (ibatch + 1)

            # store local energy
            elif obs == "local_energy" and local_energy is not None:
                data = local_energy.cpu().detach().numpy()
                if (ibatch is None) or (ibatch == 0):
                    self.observable.local_energy.append(data)
                else:
                    self.observable.local_energy[-1] = np.append(
                        self.observable.local_energy[-1], data
                    )

            # store variational parameter
            elif obs in self.wf.state_dict():
                p = self.wf.state_dict()[obs].clone()
                self.observable.__getattribute__(obs).append(
                    p.data.cpu().detach().numpy()
                )

                if obs + ".grad" in self.observable.__dict__.keys():
                    if p.grad is not None:
                        self.observable.__getattribute__(obs + ".grad").append(
                            p.grad.cpu().numpy()
                        )
                    else:
                        self.observable.__getattribute__(obs + ".grad").append(
                            torch.zeros_like(p.data).cpu().numpy()
                        )

            # store any other defined method
            elif hasattr(self.wf, obs):
                func = self.wf.__getattribute__(obs)
                data = func(pos)
                if isinstance(data, torch.Tensor):
                    data = data.cpu().detach().numpy()
                if isinstance(data, list):
                    data = np.array(data)
                if (ibatch is None) or (ibatch == 0):
                    self.observable.__getattribute__(obs).append(data)
                else:
                    self.observable.__getattribute__(obs)[-1] = np.append(
                        self.observable.__getattribute__(obs)[-1], data
                    )

    def print_observable(self, cumulative_loss: float, verbose: bool = False):
        """Print the observalbe to csreen

        Args:
            cumulative_loss (float): current loss value
            verbose (bool, optional): print all the observables. Defaults to False
        """

        for k in self.observable.__dict__.keys():
            if k == "local_energy":
                eloc = self.observable.local_energy[-1]
                e = np.mean(eloc)
                v = np.var(eloc)
                err = np.sqrt(v / len(eloc))
                log.options(style="percent").info("  energy   : %f +/- %f" % (e, err))
                log.options(style="percent").info("  variance : %f" % np.sqrt(v))

            elif verbose:
                log.options(style="percent").info(
                    k + " : ", self.observable.__getattribute__(k)[-1]
                )
                log.options(style="percent").info("loss %f" % (cumulative_loss))

    def resample(self, n: int, pos: torch.tensor) -> torch.tensor:
        """Resample the wave function

        Args:
            n (int): current epoch value
            pos (torch.tensor): positions of the walkers

        Returns:
            (torch.tensor): new positions of the walkers
        """

        if self.resampling_options.mode != "never":
            # resample the data
            if n % self.resampling_options.resample_every == 0:
                # make a copy of the pos if we update
                if self.resampling_options.mode == "update":
                    pos = (pos.clone().detach()[: self.sampler.walkers.nwalkers]).to(
                        self.device
                    )

                # start from scratch otherwise
                else:
                    pos = None

                # potentially increase the number of sampling point
                if self.resampling_options.increment["every"] is not None:
                    if n % self.resampling_options.increment["every"] == 0:
                        self.sampler.nstep += (
                            self.resampling_options.increment["factor"]
                            * self.sampler.ndecor
                        )

                # sample and update the dataset
                pos = self.sampler(self.wf.pdf, pos=pos, with_tqdm=False)

                self.dataloader.dataset = pos

            # update the weight of the loss if needed
            if self.loss.use_weight:
                self.loss.weight["psi0"] = None

        return pos

    def single_point(
        self,
        with_tqdm: Optional[bool] = True,
        batchsize: Optional[int] = None,
        hdf5_group: str = "single_point",
    ):
        """Performs a single point calculation

        Args:
            with_tqdm (bool, optional): use tqdm for samplig. Defaults to True.
            hdf5_group (str, optional): hdf5 group where to store the data.
                                        Defaults to 'single_point'.

        Returns:
            SimpleNamespace: contains the local energy, positions, ...
        """

        log.info("")
        log.info(
            "  Single Point Calculation : {nw} walkers | {ns} steps",
            nw=self.sampler.walkers.nwalkers,
            ns=self.sampler.nstep,
        )

        # check if we have to compute and store the grads
        grad_mode = torch.no_grad()
        if self.wf.kinetic == "auto":
            grad_mode = torch.enable_grad()

        if self.wf.jastrow.requires_autograd:
            grad_mode = torch.enable_grad()

        with grad_mode:
            #  get the position and put to gpu if necessary
            pos = self.sampler(self.wf.pdf, with_tqdm=with_tqdm)
            if self.wf.cuda and pos.device.type == "cpu":
                pos = pos.to(self.device)

            # compute energy/variance/error
            if batchsize is None:
                eloc = self.wf.local_energy(pos)

            else:
                nbatch = int(np.ceil(len(pos) / batchsize))

                for ibatch in range(nbatch):
                    istart = ibatch * batchsize
                    iend = min((ibatch + 1) * batchsize, len(pos))
                    if ibatch == 0:
                        eloc = self.wf.local_energy(pos[istart:iend, :])
                    else:
                        eloc = torch.cat(
                            (eloc, self.wf.local_energy(pos[istart:iend, :]))
                        )

            e, s, err = torch.mean(eloc), torch.var(eloc), self.wf.sampling_error(eloc)

            # print data
            log.options(style="percent").info(
                "  Energy   : %f +/- %f" % (e.detach().item(), err.detach().item())
            )
            log.options(style="percent").info("  Variance : %f" % s.detach().item())
            log.options(style="percent").info("  Size     : %d" % len(eloc))

            # dump data to hdf5
            obs = SimpleNamespace(
                pos=pos, local_energy=eloc, energy=e, variance=s, error=err
            )
            dump_to_hdf5(obs, self.hdf5file, root_name=hdf5_group)
            add_group_attr(self.hdf5file, hdf5_group, {"type": "single_point"})

        return obs

    def save_checkpoint(self, epoch: int, loss: float):
        """save the model and optimizer state

        Args:
            epoch (int): epoch
            loss (float): current value of the loss
        """
        filename = "checkpoint_epoch%d.pth" % epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.wf.state_dict(),
                "optimzier_state_dict": self.opt.state_dict(),
                "loss": loss,
            },
            filename,
        )

    def load_checkpoint(self, filename: str) -> Tuple[int, float]:
        """load a model/optmizer

        Args:
            filename (str): filename

        Returns:
            tuple : epoch number and loss
        """
        data = torch.load(filename)
        self.wf.load_state_dict(data["model_state_dict"])
        self.opt.load_state_dict(data["optimzier_state_dict"])
        epoch = data["epoch"]
        loss = data["loss"]
        return epoch, loss

    def _append_observable(self, key: str, data: Any):
        """Append a new data point to observable key.

        Arguments:
            key {str} -- name of the observable
            data {} -- data
        """

        if key not in self.obs_dict.keys():
            self.obs_dict[key] = []
        self.obs_dict[key].append(data)

    def sampling_traj(
        self,
        pos: Optional[torch.tensor] = None,
        with_tqdm: Optional[bool] = True,
        hdf5_group: Optional[str] = "sampling_trajectory",
    ) -> torch.tensor:
        """Compute the local energy along a sampling trajectory

        Args:
            pos (torch.tensor): positions of the walkers along the trajectory
            hdf5_group (str, optional): name of the group where to store the data.
                                        Defaults to 'sampling_trajecory'
        Returns:
            SimpleNamespace : contains energy/positions/
        """
        log.info("")
        log.info("  Sampling trajectory")

        if pos is None:
            pos = self.sampler(self.wf.pdf, with_tqdm=with_tqdm)

        ndim = pos.shape[-1]
        p = pos.view(-1, self.sampler.walkers.nwalkers, ndim)

        el = []
        rng = tqdm(p, desc="INFO:QMCTorch|  Energy  ", disable=not with_tqdm)
        for ip in rng:
            if self.wf.cuda and ip.device.type == "cpu":
                ip = ip.to(self.device)
            el.append(self.wf.local_energy(ip).cpu().detach().numpy())

        el = np.array(el).squeeze(-1)
        obs = SimpleNamespace(local_energy=np.array(el), pos=pos)
        dump_to_hdf5(obs, self.hdf5file, hdf5_group)

        add_group_attr(self.hdf5file, hdf5_group, {"type": "sampling_traj"})
        return obs

    def print_parameters(self, grad: Optional[bool] = False) -> None:
        """print parameter values

        Args:
            grad (bool, optional): also print the gradient. Defaults to False.
        """
        for p in self.wf.parameters():
            if p.requires_grad:
                if grad:
                    print(p.grad)
                else:
                    print(p)

    def optimization_step(self, lpos: torch.tensor) -> None:
        """Performs one optimization step

        Arguments:
            lpos {torch.tensor} -- positions of the walkers
        """

        if self.opt.lpos_needed:
            self.opt.step(lpos)
        else:
            self.opt.step()

    def save_traj(self, fname: str, obs: SimpleNamespace):
        """Save trajectory of geo_opt

        Args:
            fname (str): file name
        """
        f = open(fname, "w")
        xyz = obs.geometry
        natom = len(xyz[0])
        nm2bohr = 1.88973
        for snap in xyz:
            f.write("%d \n\n" % natom)
            for i, pos in enumerate(snap):
                at = self.wf.atoms[i]
                f.write(
                    "%s % 7.5f % 7.5f %7.5f\n"
                    % (at[0], pos[0] / nm2bohr, pos[1] / nm2bohr, pos[2] / nm2bohr)
                )
            f.write("\n")
        f.close()

    def run(self, nepoch: int, batchsize: Optional[int] = None, loss: str = "variance"):
        raise NotImplementedError()

    def log_data(self) -> None:
        """Log basic information about the sampler."""

        log.info("")
        log.info(" QMC Solver ")

        if self.wf is not None:
            log.info("  WaveFunction        : {0}", self.wf.__class__.__name__)
            for x in self.wf.__repr__().split("\n"):
                log.debug("   " + x)

        if self.sampler is not None:
            log.info("  Sampler             : {0}", self.sampler.__class__.__name__)
            for x in self.sampler.__repr__().split("\n"):
                log.debug("   " + x)

        if self.opt is not None:
            log.info("  Optimizer           : {0}", self.opt.__class__.__name__)
            for x in self.opt.__repr__().split("\n"):
                log.debug("   " + x)

        if self.scheduler is not None:
            log.info("  Scheduler           : {0}", self.scheduler.__class__.__name__)
            for x in self.scheduler.__repr__().split("\n"):
                log.debug("   " + x)
