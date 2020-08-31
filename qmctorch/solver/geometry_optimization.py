from torch import optim
from copy import deepcopy
from .observable import Observable
from .solver_orbital import SolverOrbital
from ..utils import save_trajectory
from .. import log


class GeoSolver():

    def __init__(self, solver, opt_geo=None):

        self.solver = solver
        self.opt_geo = opt_geo

    def run(self, nepoch, geo_lr=1e-2, batchsize=None,
            nepoch_wf_init=100, nepoch_wf_update=50,
            hdf5_group=None, chkpt_every=None):
        """optimize the geometry of the molecule

        Args:
            nepoch (int): Number of optimziation step
            batchsize (int, optional): Number of sample in a mini batch.
                                       If None, all samples are used.
                                       Defaults to Never.
            hdf5_group (str, optional): name of the hdf5 group where to store the data.
                                        Defaults to wf.task.
            chkpt_every (int, optional): save a checkpoint every every iteration.
                                         Defaults to half the number of epoch
        """

        if not hasattr(self.solver.observable, 'geometry'):
            self.solver.observable.geometry = []

        # save the optimizer used for the wf params
        opt_wf = deepcopy(self.solver.opt)

        # create the optmizier for the geo opt
        if self.opt_geo is None:
            opt_geo = optim.SGD(
                self.solver.wf.parameters(), lr=geo_lr)
        else:
            opt_geo = self.opt_geo

        # save the grad method
        eval_grad_wf = self.solver.evaluate_gradient

        # log data
        self.solver.prepare_optimization(batchsize, None)
        self.log_data_geo(nepoch)

        # init the traj
        xyz = [self.solver.wf.geometry(None)]

        # initial wf optimization
        self.solver.set_params_requires_grad(wf_params=True,
                                             geo_params=False)
        self.solver.freeze_parameters(self.solver.freeze_params_list)
        self.solver.run_epochs(nepoch_wf_init)

        # iterations over geo optim
        for n in range(nepoch):

            # make one step geo optim
            self.solver.set_params_requires_grad(wf_params=False,
                                                 geo_params=True)
            self.solver.opt = opt_geo
            self.solver.evaluate_gradient = self.solver.evaluate_grad_auto
            self.solver.run_epochs(1)
            xyz.append(self.solver.wf.geometry(None))

            # make a few wf optim
            self.solver.set_params_requires_grad(wf_params=True,
                                                 geo_params=False)
            self.solver.freeze_parameters(
                self.solver.freeze_params_list)
            self.solver.opt = opt_wf
            self.solver.evaluate_gradient = eval_grad_wf

            cumulative_loss = self.solver.run_epochs(nepoch_wf_update)

            # save checkpoint file
            if chkpt_every is not None:
                if (n > 0) and (n % chkpt_every == 0):
                    self.solver.save_checkpoint(n, cumulative_loss)

        # dump
        self.solver.observable.geometry = xyz
        self.solver.observable.save(
            hdf5_group or 'geo_opt', self.solver.hdf5file)

        # save traj
        filename = self.solver.wf.mol.name + '_go_traj.xyz'
        save_trajectory(filename, self.solver.wf.atoms, xyz)

        return self.solver.observable

    def log_data_geo(self, nepoch):
        """Log data for the optimization."""
        log.info('')
        log.info('  Optimization')
        log.info(
            '  Number Parameters   : {0}', self.solver.wf.get_number_parameters())
        log.info('  Number of epoch     : {0}', nepoch)
        log.info(
            '  Batch size          : {0}', self.solver.sampler.get_sampling_size())
        log.info(
            '  Loss function       : {0}', self.solver.loss.method)
        log.info('  Clip Loss           : {0}', self.solver.loss.clip)
        log.info(
            '  Gradients           : {0}', self.solver.grad_method)
        log.info(
            '  Resampling mode     : {0}', self.solver.resampler.options.mode)
        log.info(
            '  Resampling every    : {0}', self.solver.resampler.options.resample_every)
        log.info(
            '  Resampling steps    : {0}', self.solver.resampler.options.nstep_update)
        log.info(
            '  Output file         : {0}', self.solver.hdf5file)
        log.info(
            '  Checkpoint every    : {0}', self.solver.chkpt_every)
        log.info('')
