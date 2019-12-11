import torch
import torch.optim as optim

from deepqmc.wavefunction.wf_orbital import Orbital
from deepqmc.solver.solver_orbital import SolverOrbital

from deepqmc.sampler.metropolis import Metropolis
from deepqmc.wavefunction.molecule import Molecule

import unittest


class OrbitalH2(Orbital):

    def __init__(self, mol):
        super(OrbitalH2, self).__init__(
            mol, kinetic='jacobi', use_projector=False)

    def pool(self, x):
        return (x[:, 0, 0]*x[:, 1, 0]).view(-1, 1)


class TestH2(unittest.TestCase):

    def setUp(self):

        # optimal parameters
        self.opt_r = 0.69  # the two h are at +0.69 and -0.69
        self.opt_sigma = 1.24

        # molecule
        self.mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69', unit='bohr',
                            basis_type='gto', basis='sto-3g')

        # wave function
        self.wf = OrbitalH2(self.mol)

        # sampler
        self.sampler = Metropolis(nwalkers=1000, nstep=2000, step_size=0.5,
                                  ndim=self.wf.ndim, nelec=self.wf.nelec,
                                  init=self.mol.domain('normal'))

        # optimizer
        self.opt = optim.Adam(self.wf.parameters(), lr=0.01)

        # solver
        self.solver = SolverOrbital(wf=self.wf, sampler=self.sampler,
                                    optimizer=self.opt)

        # ground state energy
        self.ground_state_energy = -1.16

        # ground state pos
        self.ground_state_pos = 0.69

    def test_single_point(self):

        self.solver.wf.ao.atom_coords[0, 2] = -self.ground_state_pos
        self.solver.wf.ao.atom_coords[1, 2] = self.ground_state_pos

        # sample and compute observables
        _, e, v = self.solver.single_point()

        print('Energy   :', e)
        print('Variance :', v)

        # assert(e>self.ground_state_energy and e<-1.)
        assert(e > 2*self.ground_state_energy and e < 0.)
        assert(v > 0 and v < 5.)

    def test_geo_opt(self):

        self.solver.wf.ao.atom_coords[0, 2].data = torch.tensor(-0.37)
        self.solver.wf.ao.atom_coords[1, 2].data = torch.tensor(0.37)

        self.solver.configure(task='geo_opt')
        self.solver.observable(['local_energy', 'atomic_distances'])
        self.solver.run(50, loss='energy')

        # load the best model
        best_model = torch.load('model.pth')
        self.solver.wf.load_state_dict(best_model['model_state_dict'])
        self.solver.wf.eval()

        # sample and compute variables
        _, e, v = self.solver.single_point()
        e = e.data.numpy()
        v = v.data.numpy()

        # it might be too much to assert with the ground state energy
        assert(e > 2*self.ground_state_energy and e < 0.)
        assert(v > 0 and v < 2.)


if __name__ == "__main__":
    unittest.main()
