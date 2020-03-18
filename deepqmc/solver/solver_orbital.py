import torch
from torch.utils.data import DataLoader

from deepqmc.solver.solver_base import SolverBase
from deepqmc.solver.torch_utils import (DataSet, Loss,
                                        ZeroOneClipper, OrthoReg)


class SolverOrbital(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None,
                 scheduler=None):

        SolverBase.__init__(self, wf, sampler, optimizer)
        self.scheduler = scheduler
        self.ortho_mo = True

        # init sampling
        self.initial_sampling(ntherm=-1, ndecor=100)

        # resampling
        self.resampling(ntherm=-1, nstep=100,
                        resample_from_last=True,
                        resample_every=1)

        # task
        self.configure(task='geo_opt')

        # observalbe
        self.observable(['local_energy'])

        # distributed model
        self.save_model = 'model.pth'

        if self.wf.cuda:
            self.device = torch.device('cuda')
            self.sampler.cuda = True
            self.sampler.walkers.cuda = True
        else:
            self.device = torch.device('cpu')

    def configure(self, task='wf_opt', freeze=None):
        '''Configure the optimzier for specific tasks.'''

        self.task = task

        if task == 'geo_opt':
            self.wf.ao.atom_coords.requires_grad = True

            self.wf.ao.bas_coeffs.requires_grad = False
            self.wf.ao.bas_exp.requires_grad = False
            self.wf.jastrow.weight.requires_grad = False
            for param in self.wf.mo.parameters():
                param.requires_grad = False
            self.wf.fc.weight.requires_grad = False

        elif task == 'wf_opt':
            self.wf.ao.bas_exp.requires_grad = True
            self.wf.ao.bas_coeffs.requires_grad = True
            for param in self.wf.mo.parameters():
                param.requires_grad = True
            self.wf.fc.weight.requires_grad = True
            self.wf.jastrow.weight.requires_grad = True

            self.wf.ao.atom_coords.requires_grad = False

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

    def run(self, nepoch, batchsize=None,
            loss='variance',
            clip_loss=False,
            grad='auto'):
        '''Train the model.

        Arg:
            nepoch : number of epoch
            batchsize : size of the minibatch, if None take all points at once
            loss : loss used ('energy','variance' or callable (for supervised)
        '''

        if 'lpos_needed' not in self.opt.__dict__.keys():
            self.opt.lpos_needed = False

        # sample the wave function
        pos = self.sample(ntherm=self.initial_sample.ntherm,
                          ndecor=self.initial_sample.ndecor)

        # resize the number of walkers
        _nwalker_save = self.sampler.walkers.nwalkers
        if self.resample.resample_from_last:
            self.sampler.walkers.nwalkers = pos.shape[0]
            self.sampler.nwalkers = pos.shape[0]

        # handle the batch size
        if batchsize is None:
            batchsize = len(pos)

        # change the number of steps
        _nstep_save = self.sampler.nstep
        _step_size_save = self.sampler.step_size

        self.sampler.nstep = self.resample.resample
        self.sampler.step_size = self.resample.step_size

        # create the data loader
        self.dataset = DataSet(pos)
        self.dataloader = DataLoader(self.dataset, batch_size=batchsize)

        # get the loss
        self.loss = Loss(self.wf, method=loss, clip=clip_loss)

        # orthogonalization penalty for the MO coeffs
        self.ortho_loss = OrthoReg()

        # clipper for the fc weights
        self.clipper = ZeroOneClipper()

        cumulative_loss = []
        min_loss = 1E3

        # get the initial observalbe
        self.get_observable(self.obs_dict, pos)

        # loop over the epoch
        for n in range(nepoch):
            print('----------------------------------------')
            print('epoch %d' % n)

            cumulative_loss = 0

            # loop over the batches
            for ibatch, data in enumerate(self.dataloader):

                # port data to device
                lpos = data.to(self.device)

                # get the gradient
                loss, eloc = self.evaluate_gradient(grad, lpos)
                cumulative_loss += loss

                # optimize the parameters
                self.optimization_step(lpos)

                # observable
                self.get_observable(self.obs_dict, pos,
                                    local_energy=eloc, ibatch=ibatch)

            # save the model if necessary
            if cumulative_loss < min_loss:
                min_loss = self.save_checkpoint(
                    n, cumulative_loss, self.save_model)

            self.print_observable(cumulative_loss)

            print('----------------------------------------')

            # resample the data
            pos = self._resample(n, nepoch, pos)

            if self.task == 'geo_opt':
                self.wf.update_mo_coeffs()

            if self.scheduler is not None:
                self.scheduler.step()

        # restore the sampler number of step
        self.sampler.nstep = _nstep_save
        self.sampler.step_size = _step_size_save
        self.sampler.walkers.nwalkers = _nwalker_save
        self.sampler.nwalkers = _nwalker_save

    def _resample(self, n, nepoch, pos):

        if self.resample.resample_every is not None:

            # resample the data
            if (n % self.resample.resample_every == 0) or (n == nepoch-1):

                if self.resample.resample_from_last:
                    pos = pos.clone().detach().to(self.device)
                else:
                    pos = None
                pos = self.sample(
                    pos=pos, ntherm=self.resample.ntherm, with_tqdm=self.resample.tqdm)
                self.dataloader.dataset.data = pos

            # update the weight of the loss if needed
            if self.loss.use_weight:
                self.loss.weight['psi0'] = None

        return pos

    def evaluate_gradient(self, grad, lpos):

        if grad == 'auto':
            loss, eloc = self._evaluate_grad_auto(lpos)

        elif grad == 'manual':
            loss, eloc = self._evaluate_grad_manual(lpos)
        else:
            raise ValueError('Gradient method should be auto or stab')

        if torch.isnan(loss):
            raise ValueError("Nans detected in the loss")

        return loss, eloc

    def _evaluate_grad_auto(self, lpos):
        '''Evaluate the gradient using automatic diff
        of the required loss.'''

        # compute the loss
        loss, eloc = self.loss(lpos)

        # add mo orthogonalization if required
        if self.wf.mo.weight.requires_grad and self.ortho_mo:
            loss += self.ortho_loss(self.wf.mo.weight)

        # compute local gradients
        self.opt.zero_grad()
        loss.backward()

        return loss, eloc

    def _evaluate_grad_manual(self, lpos):

        if self.loss.method in ['energy', 'weighted-energy']:

            ''' Get the gradient of the total energy
            dE/dk = < (dpsi/dk)/psi (E_L - <E_L >) >
            '''

            # compute local energy and wf values
            _, eloc = self.loss(lpos, no_grad=True)
            psi = self.wf(lpos)
            norm = 1./len(psi)

            # evaluate the prefactor of the grads
            weight = eloc.clone()
            weight -= torch.mean(eloc)
            weight /= psi
            weight *= 2.
            weight *= norm

            # compute the gradients
            self.opt.zero_grad()
            psi.backward(weight)

            return torch.mean(eloc), eloc

        else:
            raise ValueError('Manual gradient only for energy min')

    def optimization_step(self, lpos):
        '''make one optimization step.'''

        if self.opt.lpos_needed:
            self.opt.step(lpos)
        else:
            self.opt.step()

        if self.wf.fc.clip:
            self.wf.fc.apply(self.clipper)

    def print_parameters(self, grad=False):
        for p in self.wf.parameters():
            if p.requires_grad:
                if grad:
                    print(p.grad)
                else:
                    print(p)

    def save_traj(self, fname):

        f = open(fname, 'w')
        xyz = self.obs_dict['geometry']
        natom = len(xyz[0])
        nm2bohr = 1.88973
        for snap in xyz:
            f.write('%d \n\n' % natom)
            for at in snap:
                f.write('%s % 7.5f % 7.5f %7.5f\n' % (at[0], at[1][0]/nm2bohr,
                                                      at[1][1]/nm2bohr,
                                                      at[1][2]/nm2bohr))
            f.write('\n')
        f.close()
