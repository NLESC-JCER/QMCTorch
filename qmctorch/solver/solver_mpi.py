from time import time
from types import SimpleNamespace
from typing import Optional, Dict, Union, List, Tuple, Any
from ..wavefunction import WaveFunction
from ..sampler import SamplerBase

import torch
from ..utils import DataLoader, add_group_attr, dump_to_hdf5
from .loss import Loss

from .. import log
from .solver import Solver

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    pass


def logd(rank: int, *args):
    if rank == 0:
        log.info(*args)


class SolverMPI(Solver):
    def __init__(        
            self,
            wf: Optional[WaveFunction] = None,
            sampler: Optional[SamplerBase] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            output: Optional[str] = None,
            rank: int = 0,
        ) -> None:
        """Distributed QMC solver

        Args:
            wf (qmctorch.WaveFunction, optional): wave function. Defaults to None.
            sampler (qmctorch.sampler, optional): Sampler. Defaults to None.
            optimizer (torch.optim, optional): optimizer. Defaults to None.
            scheduler (torch.optim, optional): scheduler. Defaults to None.
            output (str, optional): hdf5 filename. Defaults to None.
            rank (int, optional): rank of he process. Defaults to 0.
        """

        super().__init__(wf, sampler, optimizer, scheduler, output, rank)

        hvd.broadcast_optimizer_state(self.opt, root_rank=0)
        self.opt = hvd.DistributedOptimizer(
            self.opt, named_parameters=self.wf.named_parameters()
        )

        self.sampler.walkers.nwalkers //= hvd.size()

    def run(  # pylint: disable=too-many-arguments
        self,
        nepoch: int,
        batchsize: Optional[int] = None,
        loss: str = "energy",
        clip_loss: bool = False,
        grad: str = "manual",
        hdf5_group: str = "wf_opt",
        num_threads: int = 1,
        chkpt_every: Optional[int] = None,
    ) -> SimpleNamespace:
        """Run the optimization

        Args:
            nepoch (int): Number of optimization step
            batchsize (int, optional): Number of sample in a mini batch.
                                       If None, all samples are used.
                                       Defaults to None.
            loss (str, optional): method to compute the loss: variance or energy.
                                  Defaults to 'energy'.
            clip_loss (bool, optional): Clip the loss values at +/- 5std.
                                        Defaults to False.
            grad (str, optional): method to compute the gradients: 'auto' or 'manual'.
                                  Defaults to 'auto'.
            hdf5_group (str, optional): name of the hdf5 group where to store the data.
                                    Defaults to 'wf_opt'
        """

        logd(hvd.rank(), "")
        logd(
            hvd.rank(),
            "  Distributed Optimization on {num} process".format(num=hvd.size()),
        )
        log.info(
            "   - Process {id} using {nw} walkers".format(
                id=hvd.rank(), nw=self.sampler.walkers.nwalkers
            )
        )

        # observable
        if not hasattr(self, "observable"):
            self.track_observable(["local_energy"])

        self.evaluate_gradient = {
            "auto": self.evaluate_grad_auto,
            "manual": self.evaluate_grad_manual,
        }[grad]

        if "lpos_needed" not in self.opt.__dict__.keys():
            self.opt.lpos_needed = False

        self.wf.train()

        hvd.broadcast_parameters(self.wf.state_dict(), root_rank=0)
        torch.set_num_threads(num_threads)

        # get the loss
        self.loss = Loss(self.wf, method=loss, clip=clip_loss)
        self.loss.use_weight = self.resampling_options.resample_every > 1

        self.prepare_optimization(batchsize, chkpt_every)
        # log data
        if hvd.rank() == 0:
            self.log_data_opt(nepoch, "wave function optimization")

        # sample the wave function
        if hvd.rank() == 0:
            pos = self.sampler(self.wf.pdf)
        else:
            pos = self.sampler(self.wf.pdf, with_tqdm=False)

        # requried to build the distributed data container
        pos.requires_grad_(False)

        # handle the batch size
        if batchsize is None:
            batchsize = len(pos)

        # get the initial observable
        if hvd.rank() == 0:
            self.store_observable(pos)

        # change the number of steps/walker size
        _nstep_save = self.sampler.nstep
        _ntherm_save = self.sampler.ntherm
        _nwalker_save = self.sampler.walkers.nwalkers
        if self.resampling_options.mode == "update":
            self.sampler.ntherm = -1
            self.sampler.nstep = self.resampling_options.nstep_update
            self.sampler.walkers.nwalkers = pos.shape[0]

        # create the data loader
        # self.dataset = DataSet(pos)
        self.dataloader = DataLoader(pos, batch_size=batchsize, pin_memory=self.cuda)
        min_loss = 1e3

        for n in range(nepoch):
            tstart = time()
            logd(hvd.rank(), "")
            logd(hvd.rank(), "  epoch %d" % n)

            cumulative_loss = 0.0

            for ibatch, data in enumerate(self.dataloader):
                # get data
                lpos = data.to(self.device)
                lpos.requires_grad = True

                # get the gradient
                loss, eloc = self.evaluate_gradient(lpos)
                cumulative_loss += loss

                # optimize the parameters
                self.optimization_step(lpos)

                # observable
                if hvd.rank() == 0:
                    self.store_observable(pos, local_energy=eloc, ibatch=ibatch)

            cumulative_loss = self.metric_average(cumulative_loss, "cum_loss")

            if hvd.rank() == 0:
                if n == 0 or cumulative_loss < min_loss:
                    self.observable.models.best = dict(self.wf.state_dict())
                min_loss = cumulative_loss

                if self.chkpt_every is not None:
                    if (n > 0) and (n % chkpt_every == 0):
                        self.save_checkpoint(n, cumulative_loss)

                self.print_observable(cumulative_loss)

            # resample the data
            pos = self.resample(n, pos)
            pos.requires_grad = False

            # scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            logd(hvd.rank(), "  epoch done in %1.2f sec." % (time() - tstart))

        # restore the sampler number of step
        self.sampler.nstep = _nstep_save
        self.sampler.ntherm = _ntherm_save
        self.sampler.walkers.nwalkers = _nwalker_save

        if hvd.rank() == 0:
            dump_to_hdf5(self.observable, self.hdf5file, hdf5_group)
            add_group_attr(self.hdf5file, hdf5_group, {"type": "opt"})

        return self.observable

    def single_point(
        self, 
        with_tqdm: bool = True, 
        batchsize: Optional[int] = None, 
        hdf5_group: str = "single_point"
    ) -> SimpleNamespace:
        """Performs a single point calculation

        Args:
            with_tqdm (bool, optional): use tqdm for samplig. Defaults to True.
            batchsize (int, optional): Number of sample in a mini batch. If None, all samples are used.
                                      Defaults to Never.
            hdf5_group (str, optional): hdf5 group where to store the data.
                                        Defaults to 'single_point'.

        Returns:
            SimpleNamespace: contains the local energy, positions, ...
        """

        logd(hvd.rank(), "")
        logd(
            hvd.rank(),
            "  Single Point Calculation : {nw} walkers | {ns} steps".format(
                nw=self.sampler.walkers.nwalkers, ns=self.sampler.nstep
            ),
        )

        if batchsize is not None:
            log.info("  Batchsize not supported for MPI solver")

        # check if we have to compute and store the grads
        grad_mode = torch.no_grad()
        if self.wf.kinetic == "auto":
            grad_mode = torch.enable_grad()

        # distribute the calculation
        num_threads = 1
        hvd.broadcast_parameters(self.wf.state_dict(), root_rank=0)
        torch.set_num_threads(num_threads)

        with grad_mode:
            # sample the wave function
            pos: torch.tensor = self.sampler(self.wf.pdf, with_tqdm=with_tqdm)
            if self.wf.cuda and pos.device.type == "cpu":
                pos = pos.to(self.device)

            # compute energy/variance/error
            eloc: torch.tensor = self.wf.local_energy(pos)
            e: torch.tensor = torch.mean(eloc)
            s: torch.tensor = torch.var(eloc)
            err: torch.tensor = self.wf.sampling_error(eloc)

            # gather all data
            eloc_all: torch.tensor = hvd.allgather(eloc, name="local_energies")
            e = torch.mean(eloc_all)
            s = torch.var(eloc_all)
            err = self.wf.sampling_error(eloc_all)

            # print
            if hvd.rank() == 0:
                log.options(style="percent").info(
                    "  Energy   : %f +/- %f" % (e.detach().item(), err.detach().item())
                )
                log.options(style="percent").info("  Variance : %f" % s.detach().item())

            # dump data to hdf5
            obs: SimpleNamespace = SimpleNamespace(
                pos=pos, local_energy=eloc_all, energy=e, variance=s, error=err
            )

            # dump to file
            if hvd.rank() == 0:
                dump_to_hdf5(obs, self.hdf5file, root_name=hdf5_group)
                add_group_attr(self.hdf5file, hdf5_group, {"type": "single_point"})

        return obs

    @staticmethod
    def metric_average(val: torch.Tensor, name: str) -> float:
        """Average a given quantity over all processes

        Args:
            val (torch.Tensor): data to average
            name (str): name of the data

        Returns:
            float: Averaged quantity
        """
        tensor = val.clone().detach()
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()
