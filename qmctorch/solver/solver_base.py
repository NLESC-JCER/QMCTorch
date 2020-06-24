import torch
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np

from ..utils import dump_to_hdf5, add_group_attr
from .. import log


class SolverBase(object):

    def __init__(self, wf=None, sampler=None,
                 optimizer=None, scheduler=None,
                 output=None, rank=0):
        """Base Class for QMC solver 

        Args:
            wf (qmctorch.WaveFunction, optional): wave function. Defaults to None.
            sampler (qmctorch.sampler, optional): Sampler. Defaults to None.
            optimizer (torch.optim, optional): optimizer. Defaults to None.
            scheduler (torch.optim, optional): scheduler. Defaults to None.
            output (str, optional): hdf5 filename. Defaults to None.
            rank (int, optional): rank of he process. Defaults to 0.
        """

        self.wf = wf
        self.sampler = sampler
        self.opt = optimizer
        self.scheduler = scheduler
        self.cuda = False
        self.device = torch.device('cpu')
        self.task = None

        # penalty to orthogonalize the MO
        # see torch_utils.py
        self.ortho_mo = False

        # default task optimize the wave function
        self.configure(task='wf_opt')

        # resampling
        self.configure_resampling()

        # distributed model
        self.save_model = 'model.pth'

        # handles GPU availability
        if self.wf.cuda:
            self.device = torch.device('cuda')
            self.sampler.cuda = True
            self.sampler.walkers.cuda = True
        else:
            self.device = torch.device('cpu')

        self.hdf5file = output
        if output is None:
            basename = self.wf.mol.hdf5file.split('.')[0]
            self.hdf5file = basename + '_QMCTorch.hdf5'

        if rank == 0:
            dump_to_hdf5(self, self.hdf5file)

        self.log_data()

    def configure_resampling(self, mode='update', resample_every=1, nstep_update=25):
        """Configure the resampling

        Args:
            mode (str, optional): method to resample : 'full', 'update', 'never' 
                                  Defaults to 'update'.
            resample_every (int, optional): Number of optimization steps between resampling
                                 Defaults to 1.
            nstep_update (int, optional): Number of MC steps in update mode. 
                                          Defaults to 25.
        """

        self.resampling_options = SimpleNamespace()
        valid_mode = ['never', 'full', 'update']
        if mode not in valid_mode:
            raise ValueError(
                mode, 'not a valid update method : ', valid_mode)

        self.resampling_options.mode = mode
        self.resampling_options.resample_every = resample_every
        self.resampling_options.nstep_update = nstep_update

    def configure(self, task='wf_opt', freeze=None):
        """Configure the optimization.

        Args:
            task (str, optional): Optimization task: 'wf_opt', 'geo_opt'.
                                  Defaults to 'wf_opt'.
            freeze (list, optional): list pf layers to freeze.
                                     Defaults to None.
        """

        self.task = task

        if task == 'geo_opt':
            self.configure_geo_opt()

        elif task == 'wf_opt':
            self.configure_wf_opt()

            self.freeze_parameters(freeze)

    def configure_geo_opt(self):
        """Configure the solver for geometry optimization."""

        # opt atom coordinate
        self.wf.ao.atom_coords.requires_grad = True

        # no ao opt
        self.wf.ao.bas_coeffs.requires_grad = False
        self.wf.ao.bas_exp.requires_grad = False

        # no jastrow opt
        self.wf.jastrow.weight.requires_grad = False

        # no mo opt
        for param in self.wf.mo.parameters():
            param.requires_grad = False

        # no ci opt
        self.wf.fc.weight.requires_grad = False

    def configure_wf_opt(self):
        """Configure the solver for wf optimization."""

        # opt all wf parameters
        self.wf.ao.bas_exp.requires_grad = True
        self.wf.ao.bas_coeffs.requires_grad = True

        for param in self.wf.mo.parameters():
            param.requires_grad = True

        self.wf.fc.weight.requires_grad = True

        for param in self.wf.jastrow.parameters():
            param.requires_grad = True

        # no opt the atom positions
        self.wf.ao.atom_coords.requires_grad = False

    def freeze_parameters(self, freeze):
        """Freeze the optimization of specified params.

        Args:
            freeze (list): list of param to freeze
        """
        if freeze is not None:
            if not isinstance(freeze, list):
                freeze = [freeze]

            for name in freeze:
                if name.lower() == 'ci':
                    self.wf.fc.weight.requires_grad = False

                elif name.lower() == 'mo':
                    for param in self.wf.mo.parameters():
                        param.requires_grad = False

                elif name.lower() == 'ao':
                    self.wf.ao.bas_exp.requires_grad = False
                    self.wf.ao.bas_coeffs.requires_grad = False

                elif name.lower() == 'jastrow':
                    for param in self.wf.jastrow.parameters():
                        param.requires_grad = False

                else:
                    opt_freeze = ['ci', 'mo', 'ao', 'jastrow']
                    raise ValueError(
                        'Valid arguments for freeze are :', opt_freeze)

    def track_observable(self, obs_name):
        """define the observalbe we want to track

        Args:
            obs_name (list): list of str defining the observalbe.
                             Each str must correspond to a WaveFuncion method
        """

        # make sure it's a list
        if not isinstance(obs_name, list):
            obs_name = list(obs_name)

        # sanity check
        valid_obs_name = ['energy', 'local_energy',
                          'geometry', 'parameters', 'gradients']
        for name in obs_name:
            if name in valid_obs_name:
                continue
            elif hasattr(self.wf, name):
                continue
            else:
                log.info(
                    '   Error : Observable %s not recognized' % name)
                log.info('         : Possible observable')
                for n in valid_obs_name:
                    log.info('         :  - %s' % n)
                log.info(
                    '         :  - or any method of the wave function')
                raise ValueError('Observable not recognized')

        # reset the Namesapce
        self.observable = SimpleNamespace()

        # add the energy of the sytem
        if 'energy' not in obs_name:
            obs_name += ['energy']

        if self.task == 'geo_opt' and 'geometry' not in obs_name:
            obs_name += ['geometry']

        for k in obs_name:

            if k == 'parameters':
                for key, p in zip(self.wf.state_dict().keys(),
                                  self.wf.parameters()):
                    if p.requires_grad:
                        self.observable.__setattr__(key, [])

            elif k == 'gradients':
                for key, p in zip(self.wf.state_dict().keys(),
                                  self.wf.parameters()):
                    if p.requires_grad:
                        self.observable.__setattr__(key+'.grad', [])

            else:
                self.observable.__setattr__(k, [])

    def store_observable(self, pos, local_energy=None, ibatch=None, **kwargs):
        """store observale in the dictionary

        Args:
            obs_dict (dict): dictionary of the observalbe
            pos (torch.tensor): positions of th walkers
            local_energy (torch.tensor, optional): precomputed values of the local
                                           energy. Defaults to None
            ibatch (int): index of the current batch. Defaults to None
        """

        if self.wf.cuda and pos.device.type == 'cpu':
            pos = pos.to(self.device)

        for obs in self.observable.__dict__.keys():

            # store the energy
            if obs == 'energy' and local_energy is not None:
                data = local_energy.cpu().detach().numpy()
                if (ibatch is None) or (ibatch == 0):
                    self.observable.energy.append(np.mean(data))
                else:
                    self.observable.energy[-1] *= ibatch/(ibatch+1)
                    self.observable.energy[-1] += np.mean(
                        data)/(ibatch+1)

            # store local energy
            elif obs == 'local_energy' and local_energy is not None:
                data = local_energy.cpu().detach().numpy()
                if (ibatch is None) or (ibatch == 0):
                    self.observable.local_energy.append(data)
                else:
                    self.observable.local_energy[-1] = np.append(
                        self.observable.local_energy[-1], data)

            # store variational parameter
            elif obs in self.wf.state_dict():
                layer, param = obs.split('.')
                p = self.wf.__getattr__(layer).__getattr__(param)
                self.observable.__getattribute__(
                    obs).append(p.data.cpu().numpy())

                if obs+'.grad' in self.observable.__dict__.keys():
                    if p.grad is not None:
                        self.observable.__getattribute__(obs +
                                                         '.grad').append(p.grad.cpu().numpy())
                    else:
                        self.observable.__getattribute__(obs +
                                                         '.grad').append(torch.zeros_like(p.data).cpu().numpy())

            # store any other defined method
            elif hasattr(self.wf, obs):
                func = self.wf.__getattribute__(obs)
                data = func(pos)
                if isinstance(data, torch.Tensor):
                    data = data.cpu().detach().numpy()
                self.observable.__getattribute__(obs).append(data)

    def print_observable(self, cumulative_loss, verbose=False):
        """Print the observalbe to csreen

        Args:
            cumulative_loss (float): current loss value
            verbose (bool, optional): print all the observables. Defaults to False
        """

        for k in self.observable.__dict__.keys():

            if k == 'local_energy':

                eloc = self.observable.local_energy[-1]
                e = np.mean(eloc)
                v = np.var(eloc)
                err = np.sqrt(v / len(eloc))
                log.options(style='percent').info(
                    '  energy   : %f +/- %f' % (e, err))
                log.options(style='percent').info(
                    '  variance : %f' % np.sqrt(v))

            elif verbose:
                log.options(style='percent').info(
                    k + ' : ', self.observable.__getattribute__(k)[-1])
                log.options(style='percent').info(
                    'loss %f' % (cumulative_loss))

    def resample(self, n, pos):
        """Resample the wave function

        Args:
            n (int): current epoch value
            nepoch (int): total number of epoch
            pos (torch.tensor): positions of the walkers

        Returns:
            (torch.tensor): new positions of the walkers
        """

        if self.resampling_options.mode != 'never':

            # resample the data
            if (n % self.resampling_options.resample_every == 0):

                # make a copy of the pos if we update
                if self.resampling_options.mode == 'update':
                    pos = pos.clone().detach().to(self.device)

                # start from scratch otherwise
                else:
                    pos = None

                # sample and update the dataset
                pos = self.sampler(
                    self.wf.pdf, pos=pos, with_tqdm=False)
                self.dataloader.dataset.data = pos

            # update the weight of the loss if needed
            if self.loss.use_weight:
                self.loss.weight['psi0'] = None

        return pos

    def single_point(self, with_tqdm=True, hdf5_group='single_point'):
        """Performs a single point calculatin

        Args:
            hdf5_group (str, optional): hdf5 group where to store the data.
                                        Defaults to 'single_point'.

        Returns:
            SimpleNamespace: contains the local energy, positions, ...
        """

        log.info('')
        log.info('  Single Point Calculation : {nw} walkers | {ns} steps',
                 nw=self.sampler.nwalkers, ns=self.sampler.nstep)

        # check if we have to compute and store the grads
        grad_mode = torch.no_grad()
        if self.wf.kinetic == 'auto':
            grad_mode = torch.enable_grad()

        with grad_mode:

            #  get the position and put to gpu if necessary
            pos = self.sampler(self.wf.pdf, with_tqdm=with_tqdm)
            if self.wf.cuda and pos.device.type == 'cpu':
                pos = pos.to(self.device)

            # compute energy/variance/error
            el = self.wf.local_energy(pos)
            e, s, err = torch.mean(el), torch.var(
                el), self.wf.sampling_error(el)

            # print data
            log.options(style='percent').info(
                '  Energy   : %f +/- %f' % (e.detach().item(), err.detach().item()))
            log.options(style='percent').info(
                '  Variance : %f' % s.detach().item())

            # dump data to hdf5
            obs = SimpleNamespace(
                pos=pos,
                local_energy=el,
                energy=e,
                variance=s,
                error=err
            )
            dump_to_hdf5(obs,
                         self.hdf5file,
                         root_name=hdf5_group)
            add_group_attr(self.hdf5file, hdf5_group,
                           {'type': 'single_point'})

        return obs

    def save_checkpoint(self, epoch, loss, filename):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.wf.state_dict(),
            'optimzier_state_dict': self.opt.state_dict(),
            'loss': loss
        }, filename)
        return loss

    def _append_observable(self, key, data):
        """Append a new data point to observable key.

        Arguments:
            key {str} -- name of the observable
            data {} -- data
        """

        if key not in self.obs_dict.keys():
            self.obs_dict[key] = []
        self.obs_dict[key].append(data)

    def sampling_traj(self, pos=None, with_tqdm=True, hdf5_group='sampling_trajectory'):
        """Compute the local energy along a sampling trajectory

        Args:
            pos (torch.tensor): positions of the walkers along the trajectory
            hdf5_group (str, optional): name of the group where to store the data.
                                        Defaults to 'sampling_trajecory'
        Returns:
            SimpleNamespace : contains energy/positions/
        """
        log.info('')
        log.info('  Sampling trajectory')

        if pos is None:
            pos = self.sampler(self.wf.pdf, with_tqdm=with_tqdm)

        ndim = pos.shape[-1]
        p = pos.view(-1, self.sampler.nwalkers, ndim)
        el = []
        rng = tqdm(p, desc='INFO:QMCTorch|  Energy  ',
                   disable=not with_tqdm)
        for ip in rng:
            el.append(self.wf.local_energy(ip).cpu().detach().numpy())

        el = np.array(el).squeeze(-1)
        obs = SimpleNamespace(local_energy=np.array(el), pos=pos)
        dump_to_hdf5(obs,
                     self.hdf5file, hdf5_group)

        add_group_attr(self.hdf5file, hdf5_group,
                       {'type': 'sampling_traj'})
        return obs

    def print_parameters(self, grad=False):
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

    def optimization_step(self, lpos):
        """Performs one optimization step

        Arguments:
            lpos {torch.tensor} -- positions of the walkers
        """

        if self.opt.lpos_needed:
            self.opt.step(lpos)
        else:
            self.opt.step()

    def save_traj(self, fname):
        """Save trajectory of geo_opt

        Args:
            fname (str): file name
        """
        f = open(fname, 'w')
        xyz = self.obs_dict['geometry']
        natom = len(xyz[0])
        nm2bohr = 1.88973
        for snap in xyz:
            f.write('%d \n\n' % natom)
            for at in snap:
                f.write('%s % 7.5f % 7.5f %7.5f\n' % (at[0],
                                                      at[1][0] /
                                                      nm2bohr,
                                                      at[1][1] /
                                                      nm2bohr,
                                                      at[1][2] / nm2bohr))
            f.write('\n')
        f.close()

    def run(self, nepoch, batchsize=None, loss='variance'):
        raise NotImplementedError()

    def log_data(self):
        """Log basi information about the sampler."""

        log.info('')
        log.info(' QMC Solver ')

        if self.wf is not None:
            log.info(
                '  WaveFunction        : {0}', self.wf.__class__.__name__)
            for x in self.wf.__repr__().split('\n'):
                log.debug('   ' + x)

        if self.sampler is not None:
            log.info(
                '  Sampler             : {0}', self.sampler.__class__.__name__)
            for x in self.sampler.__repr__().split('\n'):
                log.debug('   ' + x)

        if self.opt is not None:
            log.info(
                '  Optimizer           : {0}', self.opt.__class__.__name__)
            for x in self.opt.__repr__().split('\n'):
                log.debug('   ' + x)

        if self.scheduler is not None:
            log.info(
                '  Scheduler           : {0}', self.scheduler.__class__.__name__)
            for x in self.scheduler.__repr__().split('\n'):
                log.debug('   ' + x)
