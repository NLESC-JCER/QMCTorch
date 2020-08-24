from tqdm import tqdm
import torch
from torch.distributions import MultivariateNormal
from time import time
from types import SimpleNamespace
from .metropolis_base import MetropolisBase
from .. import log


class Metropolis(MetropolisBase):

    def __init__(self, nwalkers, ntherm,
                 nstep=None,
                 nsample=None, ndecor=None,
                 step_size=0.2,
                 nelec=1, ndim=3,
                 init={'min': -5, 'max': 5},
                 move={'type': 'all-elec', 'proba': 'normal'},
                 cuda=False):
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
            cuda (bool, optional): turn CUDA ON/OFF. Defaults to False.


        Examples::
            >>> mol = Molecule('h2.xyz')
            >>> wf = Orbital(mol)
            >>> sampler = Metropolis(nwalkers=100, nelec=wf.nelec)
            >>> pos = sampler(wf.pdf)
        """

        MetropolisBase.__init__(self, nwalkers, nsample, nstep,
                                step_size, ntherm, ndecor,
                                nelec, ndim, init, move, cuda)

    def displacement(self, num_elec, index=None):
        """get the displacement vectors for the move

        Args:
            num_elec (int): number of elec to move
            index (torch.tensor): index of elec to move
        """
        return self.random_displacement(num_elec)

    def transisition_matrix(self):
        """computes the transitions matrix"""
        return self.data.final_density / self.data.initial_density
