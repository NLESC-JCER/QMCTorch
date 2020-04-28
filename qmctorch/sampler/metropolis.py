from tqdm import tqdm
import torch
from torch.distributions import MultivariateNormal
from .sampler_base import SamplerBase


class Metropolis(SamplerBase):

    def __init__(self, nwalkers=100,
                 nstep=1000, step_size=0.2,
                 ntherm=-1, ndecor=1,
                 nelec=1, ndim=3,
                 init={'min': -5, 'max': 5},
                 move={'type': 'all-elec', 'proba': 'normal'},
                 with_tqdm=True):
        """Metropolis Hasting generator

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
            with_tqdm (bool, optional): use tqdm progress bar. Defaults to True.

        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> wf = Orbital(mol)
            >>> sampler = Metropolis(nwalkers=100, nelec=wf.nelec)
            >>> pos = sampler(wf.pdf)
        """

        SamplerBase.__init__(self, nwalkers, nstep,
                             step_size, ntherm, ndecor,
                             nelec, ndim, init, with_tqdm)

        self.configure_move(move)

    def __call__(self, pdf, pos=None):
        """Generate a series of point using MC sampling

        Args:
            pdf (callable): probability distribution function to be sampled
            pos (torch.tensor, optional): position to start with.
                                          Defaults to None.

        Returns:
            torch.tensor: positions of the walkers
        """

        _type_ = torch.get_default_dtype()
        if _type_ == torch.float32:
            eps = 1E-7
        elif _type_ == torch.float64:
            eps = 1E-16

        if self.cuda:
            self.walkers.cuda = True
            self.device = torch.device('cuda')

        if self.ntherm >= self.nstep:
            raise ValueError('Thermalisation longer than trajectory')

        with torch.no_grad():

            if self.ntherm < 0:
                self.ntherm = self.nstep + self.ntherm

            self.walkers.initialize(pos=pos)

            fx = pdf(self.walkers.pos)

            fx[fx == 0] = eps
            pos, rate, idecor = [], 0, 0

            if self.with_tqdm:
                rng = tqdm(range(self.nstep))
            else:
                rng = range(self.nstep)

            for istep in rng:

                for id_elec in self.fixed_id_elec_list:

                    # new positions
                    Xn = self.move(pdf, id_elec)

                    # new function
                    fxn = pdf(Xn)
                    fxn[fxn == 0.] = eps
                    df = fxn / fx

                    # accept the moves
                    index = self._accept(df)

                    # acceptance rate
                    rate += index.byte().sum().float().to('cpu') / \
                        (self.nwalkers * self._move_per_iter)

                    # update position/function value
                    self.walkers.pos[index, :] = Xn[index, :]
                    fx[index] = fxn[index]
                    fx[fx == 0] = eps

                if (istep >= self.ntherm):
                    if (idecor % self.ndecor == 0):
                        pos.append(self.walkers.pos.to('cpu').clone())
                    idecor += 1

            if self.with_tqdm:
                print("Acceptance rate %1.3f %%" %
                      (rate / self.nstep * 100))

        return torch.cat(pos).requires_grad_()

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

    def move(self, pdf, id_elec):
        """Move electron one at a time in a vectorized way.

        Args:
            pdf (callable): function to sample
            id_elec (int): index f the electron to move

        Returns:
            torch.tensor: new positions of the walkers
        """
        if self.nelec == 1 or self.movedict['type'] == 'all-elec':
            return self.walkers.pos + self._move(self.nelec)

        else:

            # clone and reshape data : Nwlaker, Nelec, Ndim
            new_pos = self.walkers.pos.clone()
            new_pos = new_pos.view(self.nwalkers,
                                   self.nelec, self.ndim)

            # get indexes
            if id_elec is None:
                index = torch.LongTensor(self.nwalkers).random_(
                    0, self.nelec)
            else:
                index = torch.LongTensor(self.nwalkers).fill_(id_elec)

            # change selected data
            new_pos[range(self.nwalkers), index,
                    :] += self._move(1)

            return new_pos.view(self.nwalkers, self.nelec * self.ndim)

    def _move(self, num_elec):
        """Return a random array of length size between
        [-step_size,step_size]

        Args:
            step_size (float): boundary of the array
            size (int): number of points in the array

        Returns:
            torch.tensor: random array
        """
        if self.movedict['proba'] == 'uniform':
            d = torch.rand(
                (self.nwalkers, num_elec, self.ndim), device=self.device).view(
                self.nwalkers, num_elec * self.ndim)
            return self.step_size * (2. * d - 1.)

        elif self.movedict['proba'] == 'normal':
            displacement = self.multiVariate.sample(
                (self.nwalkers, num_elec)).to(self.device)
            return displacement.view(
                self.nwalkers, num_elec * self.ndim)

    def _accept(self, P):
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
