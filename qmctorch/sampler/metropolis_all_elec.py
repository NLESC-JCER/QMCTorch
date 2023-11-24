from tqdm import tqdm
import torch
from torch.distributions import MultivariateNormal
from time import time
from typing import Callable, Union, Dict
from .sampler_base import SamplerBase
from .. import log


class Metropolis(SamplerBase):

    def __init__(self,
                 nwalkers: int = 100,
                 nstep: int = 1000,
                 step_size: float = 0.2,
                 ntherm: int = -1,
                 ndecor: int = 1,
                 nelec: int = 1,
                 ndim: int = 3,
                 init: Dict = {'min': -5, 'max': 5},
                 move: Dict = {'proba': 'normal'},
                 logspace: bool = False,
                 cuda: bool = False):
        """Metropolis Hasting generator

        Args:
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            nstep (int, optional): Number of steps. Defaults to 1000.
            step_size (int, optional): length of the step. Defaults to 0.2.
            nelec (int, optional): total number of electrons. Defaults to 1.
            ntherm (int, optional): number of mc step to thermalize. Defaults to -1, i.e. keep ponly last position
            ndecor (int, optional): number of mc step for decorelation. Defauts to 1.
            ndim (int, optional): total number of dimension. Defaults to 3.
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
            >>> wf = SlaterJastrow(mol)
            >>> sampler = Metropolis(nwalkers=100, nelec=wf.nelec)
            >>> pos = sampler(wf.pdf)
        """

        SamplerBase.__init__(self, nwalkers, nstep,
                             step_size, ntherm, ndecor,
                             nelec, ndim, init, cuda)

        self.logspace = logspace
        self.movedict = move

        if self.movedict['proba'] == 'normal':
            _sigma = self.step_size / \
                (2 * torch.sqrt(2 * torch.log(torch.as_tensor(2.))))
            self.multiVariate = MultivariateNormal(
                torch.zeros(self.ndim), _sigma * torch.eye(self.ndim))

        self.log_data()

    def log_data(self):
        """log data about the sampler."""
        log.info('  Move type           : {0}', 'all-elec')
        log.info(
            '  Move proba          : {0}', self.movedict['proba'])

    @staticmethod
    def log_func(func):
        """Compute the negative log of  a function

        Args:
            func (callable): input function

        Returns:
            callable: negative log of the function
        """
        return lambda x: torch.log(func(x))

    def __call__(self, pdf: Callable, pos: Union[None, torch.Tensor] = None,
                 with_tqdm: bool = True) -> torch.Tensor:
        """Generate a series of point using MC sampling

        Args:
            pdf (callable): probability distribution function to be sampled
            pos (torch.tensor, optional): position to start with.
                                          Defaults to None.
            with_tqdm (bool, optional): use tqdm progress bar. Defaults to True.

        Returns:
            torch.tensor: positions of the walkers
        """

        # _type_ = torch.get_default_dtype()
        # if _type_ == torch.float32:
        #     eps = 1E-7
        # elif _type_ == torch.float64:
        #     eps = 1E-16

        if self.ntherm >= self.nstep:
            raise ValueError('Thermalisation longer than trajectory')

        with torch.no_grad():

            if self.ntherm < 0:
                self.ntherm = self.nstep + self.ntherm

            self.walkers.initialize(pos=pos)
            if self.logspace:
                fx = self.log_func(pdf)(self.walkers.pos)
            else:
                fx = pdf(self.walkers.pos)

            # fx[fx == 0] = eps
            pos, rate, idecor = [], 0, 0

            rng = tqdm(range(self.nstep),
                       desc='INFO:QMCTorch|  Sampling',
                       disable=not with_tqdm)
            tstart = time()

            for istep in rng:

                # new positions
                Xn = self.move(pdf)

                if self.logspace:
                    fxn = self.log_func(pdf)(Xn)
                    df = fxn - fx

                else:
                    # new function
                    fxn = pdf(Xn)
                    # fxn[fxn == 0.] = eps
                    df = fxn / fx

                # accept the moves
                index = self._accept(df)

                # acceptance rate
                rate += index.byte().sum().float().to('cpu') / \
                    (self.walkers.nwalkers)

                # update position/function value
                self.walkers.pos[index, :] = Xn[index, :]
                fx[index] = fxn[index]
                # fx[fx == 0] = eps

                if (istep >= self.ntherm):
                    if (idecor % self.ndecor == 0):
                        pos.append(self.walkers.pos.to('cpu').clone())
                    idecor += 1

            if with_tqdm:
                log.info(
                    "   Acceptance rate     : {:1.2f} %", (rate / self.nstep * 100))
                log.info(
                    "   Timing statistics   : {:1.2f} steps/sec.", self.nstep/(time()-tstart))
                log.info(
                    "   Total Time          : {:1.2f} sec.", (time()-tstart))

        return torch.cat(pos).requires_grad_()

    def move(self, pdf: Callable) -> torch.Tensor:
        """Move electron one at a time in a vectorized way.

        Args:
            pdf (callable): function to sample

        Returns:
            torch.tensor: new positions of the walkers
        """

        return self.walkers.pos + self._move(self.nelec)

    def _move(self, num_elec: int) -> torch.Tensor:
        """propose a move for the electrons

        Args:
            num_elec (int): number of electrons to move

        Returns:
            torch.tensor: new positions of the walkers
        """
        if self.movedict['proba'] == 'uniform':
            d = torch.rand(
                (self.walkers.nwalkers, num_elec*self.ndim), device=self.device)
            return self.step_size * (2. * d - 1.)

        elif self.movedict['proba'] == 'normal':
            displacement = self.multiVariate.sample(
                (self.walkers.nwalkers, num_elec)).to(self.device)
            return displacement.view(
                self.walkers.nwalkers, num_elec * self.ndim)

    def _accept(self, proba: torch.Tensor) -> torch.Tensor:
        """accept the move or not

        Args:
            proba (torch.tensor): probability of each move

        Returns:
            t0rch.tensor: the indx of the accepted moves
        """
        if self.logspace:
            proba[proba > 0] = 0.0
            tau = torch.log(torch.rand_like(proba))
            index = (proba - tau >= 0).reshape(-1)
            return index.type(torch.bool)
        else:
            proba[proba > 1] = 1.0
            tau = torch.rand_like(proba)
            index = (proba - tau >= 0).reshape(-1)
            return index.type(torch.bool)