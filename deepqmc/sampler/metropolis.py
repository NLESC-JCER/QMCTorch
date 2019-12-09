from deepqmc.sampler.sampler_base import SamplerBase
from tqdm import tqdm
import torch


class Metropolis(SamplerBase):

    def __init__(self, nwalkers=1000, nstep=1000, nelec=1, ndim=3,
                 step_size=3, domain={'type': 'uniform', 'min': -5, 'max': 5},
                 move='all'):
        """Metroplis Hasting sampler

        Args:
            nwalkers (int, optional): [description]. Defaults to 1000.
            nstep (int, optional): [description]. Defaults to 1000.
            nelec (int, optional): [description]. Defaults to 1.
            ndim (int, optional): [description]. Defaults to 3.
            step_size (int, optional): [description]. Defaults to 3.
            domain (dict, optional): [description].
                    Defaults to {'type': 'uniform', 'min': -5, 'max': 5}.
            move (str, optional): [description]. Defaults to 'all'.
        """

        SamplerBase.__init__(self, nwalkers, nstep, nelec,
                             ndim, step_size, domain, move)

    def generate(self, pdf, ntherm=10, ndecor=100, pos=None,
                 with_tqdm=True):
        """Generate a series of point using MC sampling

        Args:
            pdf (callable): probability distribution function to be sampled
            ntherm (int, optional): number of step before thermalization.
                                    Defaults to 10.
            ndecor (int, optional): number of steps for decorrelation.
                                    Defaults to 50.
            pos (torch.tensor, optional): position to start with.
                                          Defaults to None.
            with_tqdm (bool, optional): tqdm progress bar. Defaults to True.

        Returns:
            torch.tensor: positions of the walkers
        """
        with torch.no_grad():

            if ntherm < 0:
                ntherm = self.nstep+ntherm

            self.walkers.initialize(method=self.domain['type'], pos=pos)

            fx = pdf(self.walkers.pos)
            fx[fx == 0] = 1E-16
            pos = []
            rate = 0
            idecor = 0

            if with_tqdm:
                rng = tqdm(range(self.nstep))
            else:
                rng = range(self.nstep)

            for istep in rng:

                # new positions
                Xn = self.walkers.move(self.step_size, method=self.move)

                # new function
                fxn = pdf(Xn)
                fxn[fxn == 0.] = 1E-16
                df = (fxn/(fx)).double()

                # accept the moves
                index = self._accept(df)

                # acceptance rate
                rate += index.byte().sum().float()/self.walkers.nwalkers

                # update position/function value
                self.walkers.pos[index, :] = Xn[index, :]
                fx[index] = fxn[index]
                fx[fx == 0] = 1E-16

                if (istep >= ntherm):
                    if (idecor % ndecor == 0):
                        pos.append(self.walkers.pos.clone().detach())
                    idecor += 1

            if with_tqdm:
                print("Acceptance rate %1.3f %%" % (rate/self.nstep*100))

        return torch.cat(pos)

    def _accept(self, P):
        """accept the move or not

        Args:
            P (torch.tensor): probability of each move

        Returns:
            t0rch.tensor: the indx of the accepted moves
        """
        P[P > 1] = 1.0
        tau = torch.rand(self.nwalkers).double()
        index = (P-tau >= 0).reshape(-1)
        return index.type(torch.bool)
