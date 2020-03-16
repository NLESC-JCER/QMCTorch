from deepqmc.sampler.sampler_base import SamplerBase
from tqdm import tqdm
import torch
from torch.distributions import MultivariateNormal
from time import time


class Metropolis(SamplerBase):

    def __init__(self, nwalkers=100, nstep=1000, step_size=3,
                 nelec=1, ndim=1,
                 init={'min': -5, 'max': 5},
                 move={'type': 'one-elec', 'proba': 'uniform'},
                 wf=None):
        """Metropolis Hasting generator

        Args:
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            nstep (int, optional): Number of steps. Defaults to 1000.
            step_size (int, optional): length of the step. Defaults to 3.
            nelec (int, optional): total number of electrons. Defaults to 1.
            ndim (int, optional): total number of dimension. Defaults to 1.
            init (dict, optional): method to init the positions of the walkers.
                                   uniform : {'min': min_val, 'max': max_val}
                                   normal : {'mean' : [x,y,z],
                                             'sigma':3x3 matrix}
                                   See Molecule.domain
                                   Defaults to {'min': -5, 'max': 5}.
            move (dict, optional): method to move the electrons.
                                   'type' :
                                        'one-elec': move a single electron
                                                    per iteration
                                        'all-elec': move all electrons at
                                                    the same time
                                        'all-elec-iter': move all electrons
                                                        by iterating
                                                        through single elec
                                                        moves
                                    'proba' : 'uniform', 'normal'
                                    Defaults to {'type': 'one-elec',
                                                 'proba': 'uniform'}.
        """

        SamplerBase.__init__(self, nwalkers, nstep,
                             step_size, nelec, ndim, init, move)
        self.wf = wf

        if 'type' not in self.movedict.keys():
            print('Metroplis : Set 1 electron move by default')
            self.movedict['type'] = 'one-elec'

        if 'proba' not in self.movedict.keys():
            print('Metroplis : Set uniform trial move probability')
            self.movedict['proba'] = 'uniform'

        if self.movedict['proba'] == 'normal':
            _sigma = self.step_size / \
                (2*torch.sqrt(2*torch.log(torch.tensor(2.))))
            self.multiVariate = MultivariateNormal(
                torch.zeros(self.ndim), _sigma*torch.eye(self.ndim))

        self._move_per_iter = 1
        if self.movedict['type'] not in ['all-elec-iter']:
            raise ValueError(
                " Only 1 all-elec-iter move allowed with kalos")

        if self.movedict['type'] == 'all-elec-iter':
            self.fixed_id_elec_list = range(self.nelec)
            self._move_per_iter = self.nelec
        else:
            self.fixed_id_elec_list = [None]

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

        _type_ = torch.get_default_dtype()
        if _type_ == torch.float32:
            eps = 1E-7
        elif _type_ == torch.float64:
            eps = 1E-16

        if self.cuda:
            self.walkers.cuda = True
            self.device = torch.device('cuda')

        if ntherm >= self.nstep:
            raise ValueError('Thermalisation longer than trajectory')

        with torch.no_grad():

            if ntherm < 0:
                ntherm = self.nstep+ntherm

            # init walkers
            self.walkers.initialize(pos=pos)

            # first calculations of a0, mo and slater matrices
            self.init_kalos(self.walkers.pos)

            pos, rate, idecor = [], 0, 0

            if with_tqdm:
                rng = tqdm(range(self.nstep))
            else:
                rng = range(self.nstep)

            nup = self.wf.mol.nup
            for istep in rng:

                for id_elec in self.fixed_id_elec_list:

                    t0 = time()

                    # new positions
                    Xn = self.move(id_elec)

                    # # new function
                    ao_new, sup, sdown = self.update_kalos(Xn, id_elec)
                    R = self.get_proba(sup, sdown, id_elec)

                    # accept the moves
                    index = self._accept(R**2)

                    # acceptance rate
                    rate += index.byte().sum().float().to('cpu') / \
                        (self.nwalkers*self._move_per_iter)

                    # update position/function value
                    self.walkers.pos[index, :] = Xn[index, :]
                    self.ao[index, :, :] = ao_new[index, :, :]

                    if id_elec < nup:
                        self.isup[index, :, id_elec] /= R[index].unsqueeze(-1)
                    else:
                        self.isdown[index, :, id_elec -
                                    nup] /= R[index].unsqueeze(-1)

                if (istep >= ntherm):
                    if (idecor % ndecor == 0):
                        pos.append(self.walkers.pos.to('cpu').clone())
                    idecor += 1

            if with_tqdm:
                print("Acceptance rate %1.3f %%" % (rate/self.nstep*100))

        return torch.cat(pos)

    def init_kalos(self, pos):

        if self.wf is None:
            raise ValueError('Kalos needs a wf')

        if self.wf.configs_method != 'ground_state':
            raise ValueError('Kalos only ofr Ground state only')

        self.ao = self.wf.ao(self.walkers.pos)
        self.sup, self.sdown = self.get_slater_matrix(self.ao)
        self.isup = torch.inverse(self.sup)
        self.isdown = torch.inverse(self.sdown)

    def get_proba(self, sup, sdown, id_elec):

        nup = self.wf.mol.nup
        if id_elec < nup:
            R = torch.bmm(sup[:, id_elec, :].unsqueeze(
                1), self.isup[:, :, id_elec].unsqueeze(-1)).squeeze()

        else:

            R = torch.bmm(sdown[:, id_elec-nup, :].unsqueeze(
                1), self.isdown[:, :, id_elec-nup].unsqueeze(-1)).squeeze()

        return R

    def update_kalos(self, pos_new, id_elec):
        t0 = time()
        ao_new = self.wf.ao.update(self.ao, pos_new, id_elec)
        print('udpate ', time()-t0)
        #ao = self.wf.ao(pos_new)
        #sup, sdown = self.get_slater_matrix(ao_new)
        #ao_new = self.ao.clone()

        # print('udpate ', time()-t0)

        return self.ao, self.sup, self.sdown

    def get_slater_matrix(self, ao):

        mo = self.wf.mo(self.wf.mo_scf(ao))
        slater_mat = self.wf.pool(mo, return_matrix=True)

        sup = slater_mat[0][0]
        sdown = slater_mat[1][0]

        return sup, sdown

    def move(self, id_elec):
        """Move electron one at a time in a vectorized way.

        Args:
            pdf (callable): function to sample

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

            return new_pos.view(self.nwalkers, self.nelec*self.ndim)

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
            d = torch.rand((self.nwalkers, self.ndim), device=self.device)
            return self.step_size * (2. * d - 1.)

        elif self.movedict['proba'] == 'normal':
            displacement = self.multiVariate.sample(
                (self.nwalkers, num_elec)).to(self.device)
            return displacement.view(self.nwalkers, num_elec*self.ndim)

    def _accept(self, P):
        """accept the move or not

        Args:
            P (torch.tensor): probability of each move

        Returns:
            t0rch.tensor: the indx of the accepted moves
        """

        tau = torch.rand_like(P)
        index = (P >= tau).reshape(-1)
        return index.type(torch.bool)

    def update_ao(self, ao, pos, idelec):
        ao_new = ao.clone()
        ids, ide = (idelec)*3, (idelec+1)*3

        ao_new[:, idelec, :] = self.wf.ao(
            pos[:, ids:ide], one_elec=True).squeeze(1)
        return ao_new


if __name__ == "__main__":

    def pdf(pos):
        return torch.exp(-(pos**2).prod(1))

    sampler = Metropolis()
    pos = sampler.generate(pdf)
