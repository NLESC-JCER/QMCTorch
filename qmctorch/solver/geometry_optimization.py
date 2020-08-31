from .observable import Observable
from .solver_orbital import SolverOrbital


class GeoSolver(SolverOrbital):

    def __init__(self, wf=None, sampler=None, optimizer=None, scheduler=None, output=None, rank=0):

        SolverOrbital.__init__(self, wf, sampler,
                               optimizer, scheduler, output, rank)

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

        if not hasattr(self.observable, 'geometry'):
            self.observable.geometry = []

        # save the optimizer used for the wf params
        opt_wf = deepcopy(self.opt)

        # create the optmizier for the geo opt
        opt_geo = torch.optim.SGD(self.wf.parameters(), lr=geo_lr)

        # save the grad method
        eval_grad_wf = self.evaluate_gradient

        # log data
        self.prepare_optimization(batchsize, None)
        self.log_data_geo(nepoch)

        # init the traj
        xyz = [self.wf.geometry(None)]

        # initial wf optimization
        self.set_params_requires_grad(wf_params=True,
                                      geo_params=False)
        self.freeze_parameters(self.freeze_params_list)
        self.run_epochs(nepoch_wf_init)

        # iterations over geo optim
        for n in range(nepoch):

            # make one step geo optim
            self.set_params_requires_grad(wf_params=False,
                                          geo_params=True)
            self.opt = opt_geo
            self.evaluate_gradient = self.evaluate_grad_auto
            self.run_epochs(1)
            xyz.append(self.wf.geometry(None))

            # make a few wf optim
            self.set_params_requires_grad(wf_params=True,
                                          geo_params=False)
            self.freeze_parameters(self.freeze_params_list)
            self.opt = opt_wf
            self.evaluate_gradient = eval_grad_wf

            cumulative_loss = self.run_epochs(nepoch_wf_update)

            # save checkpoint file
            if chkpt_every is not None:
                if (n > 0) and (n % chkpt_every == 0):
                    self.save_checkpoint(n, cumulative_loss)

        # dump
        self.observable.geometry = xyz
        self.save_data(hdf5_group or 'geo_opt')

        # save traj
        filename = self.wf.mol.name + '_go_traj.xyz'
        save_trajectory(filename, self.wf.atoms, xyz)

        return self.observable

        def log_data_geo(self, nepoch):
            """Log data for the optimization."""
            log.info('')
            log.info('  Optimization')
            log.info(
                '  Number Parameters   : {0}', self.wf.get_number_parameters())
            log.info('  Number of epoch     : {0}', nepoch)
            log.info(
                '  Batch size          : {0}', self.sampler.get_sampling_size())
            log.info('  Loss function       : {0}', self.loss.method)
            log.info('  Clip Loss           : {0}', self.loss.clip)
            log.info('  Gradients           : {0}', self.grad_method)
            log.info(
                '  Resampling mode     : {0}', self.resampling_options.mode)
            log.info(
                '  Resampling every    : {0}', self.resampling_options.resample_every)
            log.info(
                '  Resampling steps    : {0}', self.resampling_options.nstep_update)
            log.info(
                '  Output file         : {0}', self.hdf5file)
            log.info(
                '  Checkpoint every    : {0}', self.chkpt_every)
            log.info('')
