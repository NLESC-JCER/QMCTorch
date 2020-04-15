import torch
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np


class SolverBase(object):

    def __init__(self, wf=None, sampler=None,
                 optimizer=None, scheduler=None):
        """Base class for the solvers

        Keyword Arguments:
            wf {WaveFunction} -- WaveFuntion object (default: {None})
            sampler {SamplerBase} -- Samppler (default: {None})
            optimizer {torch.optim} -- Optimizer (default: {None})
        """
        self.wf = wf
        self.sampler = sampler
        self.opt = optimizer
        self.scheduler = scheduler
        self.cuda = False
        self.device = torch.device('cpu')
        self.task = None
        self.obs_dict = {}

        # penalty to orthogonalize the MO
        # see torch_utils.py
        self.ortho_mo = False

        # default task optimize the wave function
        self.configure(task='wf_opt')

        # init sampling
        self.initial_sampling(ntherm=-1, ndecor=100)

        # resampling
        self.resampling(ntherm=-1, nstep=100,
                        resample_from_last=True,
                        resample_every=1)

        # observalbe
        self.observable(['local_energy'])

        # distributed model
        self.save_model = 'model.pth'

        # handles GPU availability
        if self.wf.cuda:
            self.device = torch.device('cuda')
            self.sampler.cuda = True
            self.sampler.walkers.cuda = True
        else:
            self.device = torch.device('cpu')

    def resampling(self, ntherm=-1, nstep=100, step_size=None,
                   resample_from_last=True,
                   resample_every=1, tqdm=False):
        """Configure the resampling.

        Keyword Arguments:
            ntherm {int} -- Number of MC steps needed to thermalize
                            (default: {-1})
            nstep {int} -- Number of MC step (default: {100})
            step_size {float} -- step size (if none left unchanged)
                                 (default: {None})
            resample_from_last {bool} -- Use previous positions as starting
                                         point (default: {True})
            resample_every {int} -- Number of optimization step between
                                    resampling (default: {1})
            tqdm {bool} -- use tqdm (default: {False})
        """

        self.resample = SimpleNamespace()
        self.resample.ntherm = ntherm
        self.resample.resample = nstep

        if step_size is not None:
            self.resample.step_size = step_size
        else:
            self.resample.step_size = self.sampler.step_size

        self.resample.resample_from_last = resample_from_last
        self.resample.resample_every = resample_every
        self.resample.tqdm = tqdm

    def configure(self, task='wf_opt', freeze=None):
        """Configure the solver

        Keyword Arguments:
            task {str} -- task to perform (geo_opt, wf_opt)
                          (default: {'wf_opt'})
            freeze {list} -- parameters to freeze (ao, mo, jastrow, ci)
                             (default: {None})

        Raises:
            ValueError: if freeze does not good is
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
        self.wf.jastrow.weight.requires_grad = True

        # no opt the atom positions
        self.wf.ao.atom_coords.requires_grad = False

    def freeze_parameters(self, freeze):
        """Freeze the optimization of specified params.

        Arguments:
            freeze {list} -- list of param to freeze
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
                    self.wf.jastrow.weight.requires_grad = False

                else:
                    opt_freeze = ['ci', 'mo', 'ao', 'jastrow']
                    raise ValueError(
                        'Valid arguments for freeze are :', opt_freeze)

    def initial_sampling(self, ntherm=-1, ndecor=100):
        """Configure the initial sampling

        Keyword Arguments:
            ntherm {int} -- Number of MC steps needed to thermalize
                            (default: {-1})
            ndecor {int} -- number of MC step for decorelation (default: {100})
        """

        self.initial_sample = SimpleNamespace()
        self.initial_sample.ntherm = ntherm
        self.initial_sample.ndecor = ndecor

    def observable(self, obs):
        """define the observalbe we want to track

        Arguments:
            obs {list} -- list of str defining the observalbe.
                          Each str must correspond to a WaveFuncion method
        """

        # reset the dict
        self.obs_dict = {}

        for k in obs:
            self.obs_dict[k] = []

        if 'local_energy' not in self.obs_dict:
            self.obs_dict['local_energy'] = []

        if self.task == 'geo_opt' and 'geometry' not in self.obs_dict:
            self.obs_dict['geometry'] = []

        for key, p in zip(self.wf.state_dict().keys(),
                          self.wf.parameters()):
            if p.requires_grad:
                self.obs_dict[key] = []
                self.obs_dict[key + '.grad'] = []

    def sample(self, ntherm=-1, ndecor=100, with_tqdm=True, pos=None):
        """Perform a sampling

        Keyword Arguments:
            ntherm {int} -- Number of MC step for thermalization
                            (default: {-1})
            ndecor {int} -- Number of MC step for decorelation (default: {100})
            with_tqdm {bool} -- use tqdm (default: {True})
            pos {[type]} -- initial positions of the walkers (default: {None})

        Returns:
            torch.tensor -- positions of the walkers
        """

        pos = self.sampler.generate(
            self.wf.pdf, ntherm=ntherm, ndecor=ndecor,
            with_tqdm=with_tqdm, pos=pos)
        pos.requires_grad = True
        return pos

    def _resample(self, n, nepoch, pos):
        """Resample

        Arguments:
            n {int} -- current epoch value
            nepoch {int} -- total number of epoch
            pos {torch.tensor} -- positions of the walkers

        Returns:
            {torch.tensor} -- new positions of the walkers
        """

        if self.resample.resample_every is not None:

            # resample the data
            if (n % self.resample.resample_every == 0) or (
                    n == nepoch - 1):

                if self.resample.resample_from_last:
                    pos = pos.clone().detach().to(self.device)
                else:
                    pos = None
                pos = self.sample(
                    pos=pos,
                    ntherm=self.resample.ntherm,
                    with_tqdm=self.resample.tqdm)
                self.dataloader.dataset.data = pos

            # update the weight of the loss if needed
            if self.loss.use_weight:
                self.loss.weight['psi0'] = None

        return pos

    def get_observable(self, obs_dict, pos,
                       local_energy=None, ibatch=None, **kwargs):
        """store observale in the dictionary

        Arguments:
            obs_dict {dict} -- dictionary of the observalbe
            pos {torch.tensor} -- positions of th walkers

        Keyword Arguments:
            local_energy {torch.tensor} -- precomputed values of the local
                                           energy (default: {None})
            ibatch {int]} -- index of the current batch (default: {None})
        """

        if self.wf.cuda and pos.device.type == 'cpu':
            pos = pos.to(self.device)

        for obs in self.obs_dict.keys():

            # store local energy
            if obs == 'local_energy' and local_energy is not None:
                data = local_energy.cpu().detach().numpy()

                if (ibatch is None) or (ibatch == 0):
                    self.obs_dict[obs].append(data)
                else:
                    self.obs_dict[obs][-1] = np.append(
                        self.obs_dict[obs][-1], data)

            # store variational parameter
            elif obs in self.wf.state_dict():
                layer, param = obs.split('.')
                p = self.wf.__getattr__(layer).__getattr__(param)
                self.obs_dict[obs].append(p.data.clone().numpy())

                if p.grad is not None:
                    self.obs_dict[obs +
                                  '.grad'].append(p.grad.clone().numpy())
                else:
                    self.obs_dict[obs +
                                  '.grad'].append(torch.zeros_like(p.data))

            # store any other defined method
            elif hasattr(self.wf, obs):
                func = self.wf.__getattribute__(obs)
                data = func(pos)
                if isinstance(data, torch.Tensor):
                    data = data.cpu().detach().numpy()
                self.obs_dict[obs].append(data)

    def print_observable(self, cumulative_loss, verbose=False):
        """Print the observalbe to csreen

        Arguments:
            cumulative_loss {float} -- current loss value

        Keyword Arguments:
            verbose {bool} -- print all the observables (default: {False})
        """

        for k in self.obs_dict:

            if k == 'local_energy':

                eloc = self.obs_dict['local_energy'][-1]
                e = np.mean(eloc)
                v = np.var(eloc)
                err = np.sqrt(v / len(eloc))
                print('energy   : %f +/- %f' % (e, err))
                print('variance : %f' % np.sqrt(v))

            elif verbose:
                print(k + ' : ', self.obs_dict[k][-1])
                print('loss %f' % (cumulative_loss))

    def single_point(self, pos=None, prt=True,
                     with_tqdm=True, ntherm=-1, ndecor=100,
                     no_grad=True):
        """Performs a single point calculation

        Keyword Arguments:
            pos {torch.tensor} -- positions of the walkers (default: {None})
            prt {bool} -- print the value if true (default: {True})
            with_tqdm {bool} -- use tqdm(default: {True})
            ntherm {int} -- number of MC steps for thermalisation
                            (default: {-1})
            ndecor {int} -- number of MC step for decorelation (default: {100})
            no_grad {bool} -- compute gradient (default: {True})

        Returns:
            [type] -- [description]
        """

        # check if we have to compute and store the grads
        _grad = torch.enable_grad()
        if no_grad and self.wf.kinetic != 'auto':
            _grad = torch.no_grad()

        with _grad:

            if pos is None:
                pos = self.sample(ntherm=ntherm, ndecor=ndecor,
                                  with_tqdm=with_tqdm)

            if self.wf.cuda and pos.device.type == 'cpu':
                pos = pos.to(self.device)

            e, s, err = self.wf._energy_variance_error(pos)

            if prt:
                print('Energy   : ', e.detach().item(),
                      ' +/- ', err.detach().item())
                print('Variance : ', s.detach().item())

        return pos, e, s

    def save_checkpoint(self, epoch, loss, filename):
        """Save a checkpoint file

        Arguments:
            epoch {int} -- epoch number
            loss {float} -- current loss
            filename {str} -- name of the check point file

        Returns:
            float -- loss
        """
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

    def sampling_traj(self, pos):
        """Compute the local energy along a sampling trajectory

        Arguments:
            pos {torch.tensor} -- positions of the walkers along the trajectory

        Returns:
            dict -- local energy and mean energy
        """
        ndim = pos.shape[-1]
        p = pos.view(-1, self.sampler.nwalkers, ndim)
        el = []
        for ip in tqdm(p):
            el.append(self.wf.local_energy(ip).detach().numpy())
        return {'local_energy': el, 'pos': p}

    def print_parameters(self, grad=False):
        """print the parameters to screen

        Keyword Arguments:
            grad {bool} -- also print their gradients (default: {False})
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

        Arguments:
            fname {str} -- file name
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
        """Run the optimization

        Arguments:
            nepoch {int} -- number of epoch to run for

        Keyword Arguments:
            batchsize {int} -- size of the batch. If None, entire sampling
                               points are used (default: {None})
            loss {str} -- method to compute the loss (default: {'variance'})

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError()
