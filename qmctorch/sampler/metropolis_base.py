from tqdm import tqdm
import torch
from torch.distributions import MultivariateNormal
from time import time
from types import SimpleNamespace
from .sampler_base import SamplerBase
from .. import log


class MetropolisBase(SamplerBase):

    def __init__(self, mol, nwalkers, nsample,
                 nstep, step_size,
                 ntherm, ndecor,
                 init, move, cuda):
        """Metropolis Base generator

        Args:
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            nstep (int, optional): Number of steps. Defaults to 1000.
            step_size (int, optional): length of the step. Defaults to 3.
            nelec (int, optional): total number of electrons. Defaults to 1.
            ntherm (int, optional): number of mc step to thermalize. Defaults to -1, i.e. keep ponly last position
            ndecor (int, optional): number of mc step for decorelation. Defauts to 1.
            ndim (int, optional): total number of dimension. Defaults to 1.
            init (dict, optional): method to init the positions of the walkers. See Molecule.domain()

            move (dict, optional): method to move the electrons. default('all-elec','normal') \n
                                   'type':
                                        'one-elec': move a single electron per iteration \n
                                        'all-elec': move all electrons at the same time \n
                                        'all-elec-iter': move all electrons by iterating through single elec moves \n
                                    'proba' :
                                        'uniform': uniform ina cube \n
                                        'normal': gussian in a sphere \n
            cuda (bool, optional): turn CUDA ON/OFF. Defaults to False.


        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> wf = Orbital(mol)
            >>> sampler = Metropolis(nwalkers=100, nelec=wf.nelec)
            >>> pos = sampler(wf.pdf)

        Note::
            Additional fields can be created with a dictionary:
            self.additional_field = {name : func}
            where name is a string and func a callable with prototype func(pdf, pos) -> tensor
            These additional fields will be computed and updated automatically in the __call__ functions
            see for example generalized_metropolis.py
        """

        SamplerBase.__init__(self, mol, nwalkers, nsample, nstep,
                             step_size, ntherm, ndecor, init, cuda)

        self.additional_field = None
        self.configure_move(move)
        self.log_data()

    def log_data(self):
        """log data about the sampler."""
        log.info('  Move type           : {0}', self.movedict['type'])
        log.info(
            '  Move proba          : {0}', self.movedict['proba'])

    def transition_matrix(self):
        """computes the transitions matrix"""
        raise NotImplementedError(
            'Implement a transition matrix method')

    def displacement(self, num_elec, index=None):
        raise NotImplementedError('Implement a displacement method')

    def configure_move(self, move):
        """Configure the electron moves

        Args:
            move (dict, optional): method to move the electrons. default('all-elec','normal') \n
                                   'type':
                                        'one-elec': move a single electron per iteration \n
                                        'all-elec': move all electrons at the same time \n
                                        'all-elec-iter': move all electrons by iterating through single elec moves \n
                                    'proba' :
                                        'uniform': uniform ina cube \n
                                        'normal': gussian in a sphere \n

        Raises:
            ValueError: If moves are not recognized
        """

        self.movedict = move

        if 'type' not in self.movedict.keys():
            print('Metroplis : Set 1 electron move by default')
            self.movedict['type'] = 'one-elec'

        if 'proba' not in self.movedict.keys():
            print('Metroplis : Set uniform trial move probability')
            self.movedict['proba'] = 'uniform'

        if self.movedict['proba'] == 'normal':
            _sigma = self.step_size / \
                (2 * torch.sqrt(2 * torch.log(torch.tensor(2.))))
            self.multiVariate = MultivariateNormal(
                torch.zeros(self.ndim), _sigma * torch.eye(self.ndim))

        self._move_per_iter = 1
        if self.movedict['type'] not in [
                'one-elec', 'all-elec', 'all-elec-iter']:
            raise ValueError(
                " 'type' in move should be 'one-elec','all-elec', \
                  'all-elec-iter'")

        if self.movedict['type'] == 'all-elec-iter':
            self.fixed_id_elec_list = range(self.nelec)
            self._move_per_iter = self.nelec
        else:
            self.fixed_id_elec_list = [None]

    def init_sampling_data(self, pdf):
        """Computes the data needed to stat the sampling."""

        self.data = SimpleNamespace()
        self.data.initial_density = pdf(self.walkers.pos)
        self.data.initial_density[self.data.initial_density ==
                                  0] = self.epsilon
        self.data.final_density = None

        if self.additional_field is not None:
            for name, func in self.additional_field.items():
                self.data.__setattr__(
                    'initial_' + name, func(pdf, self.walkers.pos))
                self.data.__setattr__('final_'+name, None)

    def update_sampling_data(self, index):
        """Update the data for th sampling process

        Args:
            index (torch tensor): indices of the accepted move
        """

        self.walkers.pos[index, :] = self.data.final_pos[index, :]
        self.data.initial_density[index] = self.data.final_density[index]
        self.data.initial_density[self.data.initial_density ==
                                  0] = self.epsilon

        if self.additional_field is not None:
            for name, func in self.additional_field.items():
                self.data.__getattribute__(
                    'initial_' + name)[index, :] = self.data.__getattribute__(
                    'final_' + name)[index, :]

    def propose_move(self, pdf, id_elec):
        """propose a new move and computes the data

        Args:
            id_elec (torch.tensor): indexes of the elecs to move
        """

        # get new pos data
        self.data.final_pos = self.move(id_elec)

        # new densities
        self.data.final_density = pdf(self.data.final_pos)
        self.data.final_density[self.data.final_density ==
                                0.] = self.epsilon

        # additional field
        if self.additional_field is not None:
            for name, func in self.additional_field.items():
                self.data.__setattr__(
                    'final_' + name, func(pdf, self.data.final_pos))

    def move(self, id_elec):
        """Move electron one at a time in a vectorized way.

        Args:
            id_elec (int): index f the electron to move

        Returns:
            torch.tensor: new positions of the walkers
        """
        # all elec moves
        if self.nelec == 1 or self.movedict['type'] == 'all-elec':
            displacement = self.displacement(self.nelec)
            return self.walkers.pos + displacement

        # single elec moves
        else:
            index = self.get_single_electron_move_index(id_elec)
            displacement = self.displacement(1, index)
            new_pos = self.walkers.pos.clone().view(self.nwalkers,
                                                    self.nelec, self.ndim)

            new_pos[range(self.nwalkers), index, :] += displacement

            return new_pos.view(self.nwalkers, self.nelec * self.ndim)

    def get_single_electron_move_index(self, id_elec):
        """Define the index of the elec to move

        Args:
            id_elec ([type]): [description]
        """
        if id_elec is None:
            return torch.LongTensor(self.nwalkers).random_(
                0, self.nelec)
        else:
            return torch.LongTensor(self.nwalkers).fill_(id_elec)

    def accept_reject(self, P):
        """accept the move or not

        Args:
            P (torch.tensor): probability of each move

        Returns:
            t0rch.tensor: the indx of the accepted moves
        """

        P[P > 1] = 1.0
        tau = torch.rand_like(P)
        index = (P - tau >= 0).reshape(-1)
        return index.type(torch.bool)

    def random_displacement(self, num_elec):
        """get the displacement vectors for the move

        Args:
            num_elec (int): number of elec to move
            index (torch.tensor): index of elec to move
        """

        if self.movedict['proba'] == 'uniform':
            displacement = torch.rand(
                (self.nwalkers, num_elec, self.ndim), device=self.device).view(
                self.nwalkers, num_elec * self.ndim)
            displacement = self.step_size * (2. * displacement - 1.)

        elif self.movedict['proba'] == 'normal':
            displacement = self.multiVariate.sample(
                (self.nwalkers, num_elec)).to(self.device)
            displacement = displacement.view(
                self.nwalkers, num_elec * self.ndim)

        return displacement

    def __call__(self, pdf, pos=None, with_tqdm=True):
        """Generate a series of point using MC sampling

        Args:
              pdf (callable): probability distribution function to be sampled
              pos (torch.tensor, optional): position to start with.
                                            Defaults to None.
              with_tqdm (bool, optional): use tqdm progress bar. Defaults to True.

        Returns:
              torch.tensor: positions of the walkers
        """

        if self.ntherm > self.nstep:
            raise ValueError(
                'Thermalisation longer than trajectory')

        with torch.no_grad():

            if self.ntherm < 0:
                self.ntherm = self.nstep + self.ntherm

            self.walkers.initialize(pos=pos)

            if with_tqdm:
                log.info('')
                log.info('  Sampling                 : {nw} walkers | {ns} steps | {nsamp} samples ',
                         nw=self.nwalkers, ns=self.nstep, nsamp=self.nsample)

            self.init_sampling_data(pdf)

            pos, rate, idecor = [], 0, 0
            rng = tqdm(range(self.nstep),
                       desc='INFO:QMCTorch|  MC Sampling',
                       disable=not with_tqdm)
            tstart = time()

            for istep in rng:

                for id_elec in self.fixed_id_elec_list:

                    # new positions
                    self.propose_move(pdf, id_elec)

                    # get transition matrix
                    trans_mat = self.transisition_matrix()

                    # accept the moves
                    index = self.accept_reject(trans_mat)

                    # acceptance rate
                    rate += index.byte().sum().float().to('cpu') / \
                        (self.nwalkers * self._move_per_iter)

                    # update position/function value
                    # new function
                    self.update_sampling_data(index)

                if ((istep+1) >= self.ntherm):
                    if (idecor % self.ndecor == 0):
                        pos.append(
                            self.walkers.pos.to('cpu').clone())
                    idecor += 1

            if with_tqdm:
                log.info(
                    "   Acceptance rate     : {:1.2f} %", (rate / self.nstep * 100))
                log.info(
                    "   Timing statistics   : {:1.2f} steps/sec.", self.nstep/(time()-tstart))
                log.info(
                    "   Total Time          : {:1.2f} sec.", (time()-tstart))

        return torch.cat(pos).requires_grad_()
