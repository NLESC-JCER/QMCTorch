from time import time
from tqdm import tqdm
from types import SimpleNamespace
from typing import Optional, Dict, List, Tuple, Any
import torch
from ..wavefunction import WaveFunction
from ..sampler import SamplerBase
from ..utils import add_group_attr, dump_to_hdf5, DataLoader
from qmctorch.utils import add_group_attr, dump_to_hdf5, DataLoader

from .. import log
from .solver_base import SolverBase
from .loss import Loss


class Solver(SolverBase):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        wf: Optional[WaveFunction] = None,
        sampler: Optional[SamplerBase] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        output: Optional[str] = None,
        rank: int = 0,
    ) -> None:
        """Basic QMC solver

        Args:
            wf (qmctorch.WaveFunction, optional): wave function. Defaults to None.
            sampler (qmctorch.sampler, optional): Sampler. Defaults to None.
            optimizer (torch.optim, optional): optimizer. Defaults to None.
            scheduler (torch.optim, optional): scheduler. Defaults to None.
            output (str, optional): hdf5 filename. Defaults to None.
            rank (int, optional): rank of he process. Defaults to 0.
        """
        SolverBase.__init__(self, wf, sampler, optimizer, scheduler, output, rank)

        self.set_params_requires_grad()

        self.configure(
            track=["local_energy"],
            freeze=None,
            loss="energy",
            grad="manual",
            ortho_mo=False,
            clip_loss=False,
            resampling={"mode": "update", "resample_every": 1, "nstep_update": 25},
        )

    def configure(
        self,
        track: Optional[List[str]] = None,
        freeze: Optional[List[torch.nn.Parameter]] = None,
        loss: Optional[str] = None,
        grad: Optional[str] = None,
        ortho_mo: Optional[bool] = None,
        clip_loss: bool = False,
        clip_threshold: int = 5,
        resampling: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Configure the solver

        Args:
            track (list, optional): list of observable to track. Defaults to ['local_energy'].
            freeze (list, optional): list of parameters to freeze. Defaults to None.
            loss (str, optional): method to compute the loss: variance or energy.
                                  Defaults to 'energy'.
            grad (str, optional): method to compute the gradients: 'auto' or 'manual'.
                                  Defaults to 'auto'.
            ortho_mo (bool, optional): apply regularization to orthogonalize the MOs.
                                       Defaults to False.
            clip_loss (bool, optional): Clip the loss values at +/- X std. X defined in Loss
                                        as clip_num_std (default 5)
                                        Defaults to False.
            resampling (dict, optional): resampling options.
        """

        # set the parameters we want to optimize/freeze
        self.set_params_requires_grad()
        self.freeze_params_list = freeze
        self.freeze_parameters(freeze)

        # track the observable we want
        if track is not None:
            self.track_observable(track)

        # define the grad calulation
        if grad is not None:
            self.grad_method = grad
            self.evaluate_gradient = {
                "auto": self.evaluate_grad_auto,
                "manual": self.evaluate_grad_manual,
            }[grad]

        # resampling of the wave function
        if resampling is not None:
            self.configure_resampling(**resampling)

        # get the loss
        if loss is not None:
            self.loss = Loss(
                self.wf, method=loss, clip=clip_loss, clip_threshold=clip_threshold
            )
            self.loss.use_weight = self.resampling_options.resample_every > 1

        # orthogonalization penalty for the MO coeffs
        self.ortho_mo = ortho_mo
        if self.ortho_mo is True:
            log.warning(
                "Orthogonalization of the MO coeffs via loss penalty is deprecated"
            )

    def set_params_requires_grad(
        self, wf_params: Optional[bool] = True, geo_params: Optional[bool] = False
    ):
        """Configure parameters for wf opt."""

        # opt all wf parameters
        self.wf.ao.bas_exp.requires_grad = wf_params
        self.wf.ao.bas_coeffs.requires_grad = wf_params

        for param in self.wf.mo.parameters():
            param.requires_grad = wf_params

        self.wf.fc.weight.requires_grad = wf_params

        if hasattr(self.wf, "jastrow"):
            if self.wf.jastrow is not None:
                for param in self.wf.jastrow.parameters():
                    param.requires_grad = wf_params

        # no opt the atom positions
        self.wf.ao.atom_coords.requires_grad = geo_params

    def freeze_parameters(self, freeze: List[str]) -> None:
        """Freeze the optimization of specified params.

        Args:
            freeze (list): list of param to freeze
        """
        if freeze is not None:
            if not isinstance(freeze, list):
                freeze = [freeze]

            for name in freeze:
                if name.lower() == "ci":
                    self.wf.fc.weight.requires_grad = False

                elif name.lower() == "mo":
                    for param in self.wf.mo.parameters():
                        param.requires_grad = False

                elif name.lower() == "ao":
                    self.wf.ao.bas_exp.requires_grad = False
                    self.wf.ao.bas_coeffs.requires_grad = False

                elif name.lower() == "jastrow":
                    for param in self.wf.jastrow.parameters():
                        param.requires_grad = False

                elif name.lower() == "backflow":
                    for param in self.wf.ao.backflow_trans.parameters():
                        param.requires_grad = False

                else:
                    opt_freeze = ["ci", "mo", "ao", "jastrow", "backflow"]
                    raise ValueError("Valid arguments for freeze are :", opt_freeze)

    def save_sampling_parameters(self) -> None:
        """save the sampling params."""
        self.sampler._nstep_save = self.sampler.nstep
        self.sampler._ntherm_save = self.sampler.ntherm
        # self.sampler._nwalker_save = self.sampler.walkers.nwalkers

        if self.resampling_options.mode == "update":
            self.sampler.ntherm = self.resampling_options.ntherm_update
            self.sampler.nstep = self.resampling_options.nstep_update
            # self.sampler.walkers.nwalkers = pos.shape[0]

    def restore_sampling_parameters(self) -> None:
        """restore sampling params to their original values."""
        self.sampler.nstep = self.sampler._nstep_save
        self.sampler.ntherm = self.sampler._ntherm_save
        # self.sampler.walkers.nwalkers = self.sampler._nwalker_save

    def run(
        self,
        nepoch: int,
        batchsize: Optional[int] = None,
        hdf5_group: Optional[str] = "wf_opt",
        chkpt_every: Optional[int] = None,
        tqdm: Optional[bool] = False,
    ) -> SimpleNamespace:
        """Run a wave function optimization

        Args:
            nepoch (int): Number of optimziation step
            batchsize (int, optional): Number of sample in a mini batch.
                                       If None, all samples are used.
                                       Defaults to Never.
            hdf5_group (str, optional): name of the hdf5 group where to store the data.
                                        Defaults to 'wf_opt'.
            chkpt_every (int, optional): save a checkpoint every every iteration.
                                         Defaults to half the number of epoch
        """
        # prepare the optimization
        self.prepare_optimization(batchsize, chkpt_every, tqdm)
        self.log_data_opt(nepoch, "wave function optimization")

        # run the epochs
        self.run_epochs(nepoch)

        # restore the sampler number of step
        self.restore_sampling_parameters()

        # dump
        self.save_data(hdf5_group)

        return self.observable

    def prepare_optimization(
        self, batchsize: int, chkpt_every: int, tqdm: Optional[bool] = False
    ):
        """Prepare the optimization process

        Args:
            batchsize (int or None): batchsize
            chkpt_every (int or none): save a chkpt file every
        """
        log.info("  Initial Sampling    :")
        tstart = time()

        # sample the wave function
        pos = self.sampler(self.wf.pdf, with_tqdm=tqdm)

        # handle the batch size
        if batchsize is None:
            batchsize = len(pos)

        # change the number of steps/walker size
        self.save_sampling_parameters()

        # create the data loader
        self.dataloader = DataLoader(pos, batch_size=batchsize, pin_memory=self.cuda)

        for ibatch, data in enumerate(self.dataloader):
            self.store_observable(data, ibatch=ibatch)

        # chkpt
        self.chkpt_every = chkpt_every

        log.info("  done in %1.2f sec." % (time() - tstart))

    def save_data(self, hdf5_group: str):
        """Save the data to hdf5.

        Args:
            hdf5_group (str): name of group in the hdf5 file
        """
        self.observable.models.last = dict(self.wf.state_dict())

        hdf5_group = dump_to_hdf5(self.observable, self.hdf5file, hdf5_group)

        add_group_attr(self.hdf5file, hdf5_group, {"type": "opt"})

    def run_epochs(
        self,
        nepoch: int,
        with_tqdm: Optional[bool] = False,
        verbose: Optional[bool] = True,
    ) -> float:
        """Run a certain number of epochs

        Args:
            nepoch (int): number of epoch to run
        """

        if with_tqdm and verbose:
            raise ValueError("tqdm and verbose are mutually exclusive")

        # init the loss in case we have nepoch=0
        cumulative_loss = 0
        min_loss = 0  # this is set at n=0

        # the range
        rng = tqdm(
            range(nepoch),
            desc="INFO:QMCTorch|  Optimization",
            disable=not with_tqdm,
        )

        # loop over the epoch
        for n in rng:
            if verbose:
                tstart = time()
                log.info("")
                log.info(
                    "  epoch %d | %d sampling points"
                    % (n, len(self.dataloader.dataset))
                )

            # reset the gradients and loss
            cumulative_loss = 0
            self.opt.zero_grad()
            self.wf.zero_grad()

            # loop over the batches
            for ibatch, data in enumerate(self.dataloader):
                # port data to device
                lpos = data.to(self.device)

                # get the gradient
                loss, eloc = self.evaluate_gradient(lpos)
                cumulative_loss += loss.item()

                # check for nan
                if torch.isnan(eloc).any():
                    log.info("Error : Nan detected in local energy")
                    return cumulative_loss

                # observable
                self.store_observable(lpos, local_energy=eloc, ibatch=ibatch)

            # optimize the parameters
            self.optimization_step(lpos)

            # save the model if necessary
            if n == 0 or cumulative_loss < min_loss:
                min_loss = cumulative_loss
                self.observable.models.best = dict(self.wf.state_dict())

            # save checkpoint file
            if self.chkpt_every is not None:
                if (n > 0) and (n % self.chkpt_every == 0):
                    self.save_checkpoint(n, cumulative_loss)

            if verbose:
                self.print_observable(cumulative_loss, verbose=False)

            # resample the data
            self.dataloader.dataset = self.resample(n, self.dataloader.dataset)

            # scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            if verbose:
                log.info("  epoch done in %1.2f sec." % (time() - tstart))

        return cumulative_loss

    def evaluate_grad_auto(
        self, lpos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the gradient using automatic differentiation

        Args:
            lpos (torch.tensor): sampling points

        Returns:
            tuple: loss values and local energies
        """

        # compute the loss
        loss, eloc = self.loss(lpos)

        # compute local gradients
        loss.backward()

        return loss, eloc

    def evaluate_grad_manual(
        self, lpos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the gradient using low variance expression
        WARNING : This method is not valid to compute forces
        as it does not include derivative of the hamiltonian
        wrt atomic positions

        The gradients are here evaluated following:

        .. math:
            dE/dk = < (dpsi/dk)/psi (E_L - <E_L >)>

        Other estimators are possible:

        .. math:
            dE/dk = 2 [ < (dpsi/dk) E_L/psi >  - < (dpsi/dk) / psi > <E_L > ]

        given in https://www.cond-mat.de/events/correl19/manuscripts/luechow.pdf eq. 17.
        Or

        .. math:
            dE/dk = <  (E_L - <E_L >) d[ln(abs(psi))] / dk) >

        used in PauliNet


        Args:
            lpos (torch.tensor): sampling points

        Returns:
            tuple: loss values and local energies
        """

        if self.loss.method not in ["energy", "weighted-energy"]:
            raise ValueError("Manual gradient only for energy minimization")

        # compute local energy
        with torch.no_grad():
            eloc = self.wf.local_energy(lpos)

        # compute the wf values
        psi = self.wf(lpos)
        norm = 1.0 / len(psi)

        # evaluate the prefactor of the grads
        weight = eloc.clone()
        weight -= torch.mean(eloc)
        weight /= psi.clone()
        weight *= 2.0 * norm

        # clip the values
        clip_mask = self.loss.get_clipping_mask(eloc)
        psi = psi[clip_mask]
        weight = weight[clip_mask]

        # compute the gradients
        psi.backward(weight)

        return torch.mean(eloc), eloc

    def compute_forces(
        self, lpos: torch.tensor, batch_size: int = None, clip: int = None
    ) -> torch.tensor:
        r"""
        Compute the forces using automatic differentation and stable estimator

        ..math::
            F = -\\langle \\nabla_\\alpha E_L(R) + (E_L(R) - E) \\nabla)\\alpha |\Psi(R)|^2 \\rangle

        see e.g. https://arxiv.org/abs/2404.09755

        Args:
            lpos (torch.tensor): sampling points
            batch_size (int): the size of the batch to use for the automatic differentiation
            clip (int): the number of decimal places to clip the sampling points

        Returns:
            torch.tensor: the numerical forces

        """

        def get_clipping_mask(values: torch.tensor, clip: int) -> torch.tensor:
            """
            Compute a mask to clip the values based on their zscore

            Parameters
            ----------
            values : torch.tensor
                the values to clip
            clip : int
                the number of decimal places to clip the values

            Returns
            -------
            mask : torch.tensor
                the mask to clip the values
            """
            if clip is not None:
                median = torch.median(values)
                std = torch.std(values)
                zscore = torch.abs((values - median) / std)
                mask = zscore < clip
            else:
                mask = torch.ones_like(values).type(torch.bool)

            return mask

        # save the grad status of the ao
        original_requires_grad = self.wf.ao.atom_coords.requires_grad
        if not original_requires_grad:
            self.wf.ao.atom_coords.requires_grad = True

        if batch_size is None:
            batch_size = lpos.shape[0]
        nbatch = lpos.shape[0] // batch_size

        forces = torch.zeros_like(self.wf.ao.atom_coords).requires_grad_(False)
        for ibatch in range(nbatch):
            # get the batch
            idx_start = ibatch * batch_size
            idx_end = (ibatch + 1) * batch_size
            if idx_end > lpos.shape[0]:
                idx_end = lpos.shape[0]
            lpos_batch = lpos[idx_start:idx_end]

            # compute the local energy and its gradient
            local_energy = self.wf.local_energy(lpos_batch)
            clip_mask = get_clipping_mask(local_energy, clip)
            grad_eloc = torch.autograd.grad(
                local_energy, self.wf.ao.atom_coords, grad_outputs=clip_mask
            )[0]

            # compute the log density and its gradient
            wf_val = self.wf.pdf(lpos_batch)
            proba = torch.log(wf_val)
            grad_outputs = ((local_energy - local_energy.mean()) * clip_mask).squeeze()
            grad_proba = torch.autograd.grad(
                proba, self.wf.ao.atom_coords, grad_outputs=grad_outputs
            )[0]

            # accumulate in the force
            forces += 1.0 / batch_size * (grad_eloc + grad_proba)

        if not original_requires_grad:
            self.wf.ao.atom_coords.requires_grad = False

        return forces

    def log_data_opt(self, nepoch, task):
        """Log data for the optimization."""
        log.info("")
        log.info("  Optimization")
        log.info("  Task                :", task)
        log.info("  Number Parameters   : {0}", self.wf.get_number_parameters())
        log.info("  Number of epoch     : {0}", nepoch)
        log.info("  Batch size          : {0}", self.sampler.get_sampling_size())
        log.info("  Loss function       : {0}", self.loss.method)
        log.info("  Clip Loss           : {0}", self.loss.clip)
        log.info("  Gradients           : {0}", self.grad_method)
        log.info("  Resampling mode     : {0}", self.resampling_options.mode)
        log.info("  Resampling every    : {0}", self.resampling_options.resample_every)
        log.info("  Resampling steps    : {0}", self.resampling_options.nstep_update)
        log.info("  Output file         : {0}", self.hdf5file)
        log.info("  Checkpoint every    : {0}", self.chkpt_every)
        log.info("")
