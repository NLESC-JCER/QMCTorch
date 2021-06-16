import unittest
import torch
import numpy as np
from types import SimpleNamespace
from qmctorch.wavefunction.orbitals.norm_orbital import atomic_orbital_norm


class TestAtomicOrbitalNorm(unittest.TestCase):

    def test_sph_sto(self):

        basis = SimpleNamespace()
        basis.harmonics_type = 'sph'
        basis.radial_type = 'sto'
        basis.bas_n = torch.as_tensor([0, 1, 2])
        basis.bas_exp = torch.rand(3)

        atomic_orbital_norm(basis)

    def test_sph_gto(self):

        basis = SimpleNamespace()
        basis.harmonics_type = 'sph'
        basis.radial_type = 'gto'
        basis.bas_n = torch.as_tensor([0, 1, 2])
        basis.bas_exp = torch.rand(3)

        atomic_orbital_norm(basis)

    def test_cart_sto(self):

        basis = SimpleNamespace()
        basis.harmonics_type = 'cart'
        basis.radial_type = 'sto'
        basis.bas_exp = np.random.rand(4)
        basis.bas_kx = np.array([0, 0, 0, 1])
        basis.bas_ky = np.array([0, 1, 0, 0])
        basis.bas_kz = np.array([0, 0, 1, 0])
        basis.bas_kr = np.array([0, 0, 0, 0])

        atomic_orbital_norm(basis)

    def test_cart_gto(self):

        basis = SimpleNamespace()
        basis.harmonics_type = 'cart'
        basis.radial_type = 'gto'
        basis.bas_exp = np.random.rand(4)
        basis.bas_kx = np.array([0, 0, 0, 1])
        basis.bas_ky = np.array([0, 1, 0, 0])
        basis.bas_kz = np.array([0, 0, 1, 0])
        basis.bas_kr = np.array([0, 0, 0, 0])

        atomic_orbital_norm(basis)
