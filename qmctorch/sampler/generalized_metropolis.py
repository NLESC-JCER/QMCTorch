from tqdm import tqdm
import torch
from torch.autograd import Variable, grad
from torch.distributions import MultivariateNormal
import numpy as np

from .metropolis_base import MetropolisBase
from .. import log


class GeneralizedMetropolis(MetropolisBase):

    def __init__(self, mol, nwalkers, ntherm,
                 nstep=None,
                 nsample=None, ndecor=None,
                 step_size=0.2,
                 init='atomic',
                 move={'type': 'all-elec', 'proba': 'normal'},
                 cuda=False):
        """Generalized Metropolis Hasting generator

        Args:
            mol (Molecule instance): instance of a Molecule object
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            ntherm (int): number of mc step to thermalize.
            nsample (int, optional): total number of sample required
            nstep (int, optional): Number of steps. Defaults to 1000.
            step_size (int, optional): length of the step. Defaults to 3.
            ndecor (int, optional): number of mc step for decorelation. Defauts to 1.
            init (str, optional): method to init the positions of the walkers. See Molecule.domain()

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
            >>> sampler = GeneralizedMetropolis(mol, nwalkers=100, ntherm=1000)
            >>> pos = sampler(wf.pdf)
        """

        MetropolisBase.__init__(self, mol, nwalkers, nsample, nstep,
                                step_size, ntherm, ndecor, init, move, cuda)

        self.tau = self.step_size**2
        self.additional_field = {'drift': self.get_drift}

    def transisition_matrix(self):
        """computes the transitions matrix"""

        def trans(xf, xi, drifti):
            """transition probability. 
            Note we no not need the normalization factor
            as we then compute Tif/Tfi."""
            a = (xf - xi - drifti * self.tau).norm(dim=1)
            return torch.exp(- 0.5 * a / self.tau)

        Tif = trans(self.walkers.pos,
                    self.data.final_pos, self.data.final_drift)
        Tfi = trans(self.data.final_pos,
                    self.walkers.pos, self.data.initial_drift)

        return (Tif * self.data.final_density) / \
            (Tfi * self.data.initial_density)

    def displacement(self, num_elec, index=None):
        """get the displacement vectors for the move

        Args:
            num_elec (int): number of elec to move
            index (torch.tensor): index of elec to move
        """

        eps = self.random_displacement(num_elec)

        if index is None:
            displacement = eps + self.tau * self.data.initial_drift

        else:

            drift = self.data.initial_drift.view(self.nwalkers,
                                                 self.nelec, self.ndim)
            displacement = torch.zeros((
                self.nwalkers, num_elec, self.ndim)).to(self.device)

            idx_walkers = range(self.nwalkers)
            displacement[:, 0, :] += drift[idx_walkers,
                                           index, :] + eps
            displacement = displacement.view(
                self.nwalkers, num_elec*self.ndim)

        return displacement

    def get_drift(self, pdf, pos):
        """computes the drift velocity

        Args:
            pdf (callable): density function
            pos (torch.tensor): positions

        Returns:
            torch.tensor: drift velocity

        Note:
            v(R) = \grad\Psi / \Psi = 0.5 \grad |\Psi|^2 / |\Psi|^2
        """
        pdfvals, gradvals = pdf(pos, return_grad=True)
        return 0.5 * gradvals / pdfvals.unsqueeze(-1)
