import unittest
import torch
from qmctorch.sampler.symmetry import planar_symmetry, Cinfv, Dinfh


class TestPlanarSymmetry(unittest.TestCase):
    def test_single_plane(self):
        pos = torch.tensor([[1, 2, 3, 4, 5, 6]]).type(torch.float32)
        plane = "xy"
        nelec = 2
        ndim = 3
        expected_out = torch.tensor([[1, 2, -3, 4, 5, -6]]).type(torch.float32)
        out = planar_symmetry(pos, plane, nelec, ndim)
        self.assertTrue(torch.allclose(out, expected_out))

    def test_multiple_planes(self):
        pos = torch.tensor([[1, 2, 3, 4, 5, 6]]).type(torch.float32)
        plane = ["xy", "xz"]
        nelec = 2
        ndim = 3
        expected_out = torch.tensor([[1, -2, -3, 4, -5, -6]]).type(torch.float32)
        out = planar_symmetry(pos, plane, nelec, ndim)
        self.assertTrue(torch.allclose(out, expected_out))

    def test_inplace(self):
        pos = torch.tensor([[1, 2, 3, 4, 5, 6]]).type(torch.float32)
        plane = "xy"
        nelec = 2
        ndim = 3
        expected_out = torch.tensor([[1, 2, -3, 4, 5, -6]]).type(torch.float32)
        out = planar_symmetry(pos, plane, nelec, ndim, inplace=True)
        self.assertTrue(torch.allclose(out, expected_out))

    def test_invalid_plane(self):
        pos = torch.tensor([[1, 2, 3, 4, 5, 6]]).type(torch.float32)
        plane = "invalid"
        nelec = 2
        ndim = 3
        with self.assertRaises(KeyError):
            planar_symmetry(pos, plane, nelec, ndim)


class TestDinfh(unittest.TestCase):
    def setUp(self):
        self.symmetry = Dinfh("x")  # Initialize Dinfh symmetry
        self.pos = torch.randn(1, 6)  # Initialize pos tensor

    def test_valid_input(self):
        output = self.symmetry(self.pos)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (8, 6))  # Check shape of output


class TestCinfv(unittest.TestCase):
    def setUp(self):
        self.symmetry = Cinfv("x")  # Initialize Dinfh symmetry
        self.pos = torch.randn(1, 6)  # Initialize pos tensor

    def test_valid_input(self):
        output = self.symmetry(self.pos)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (4, 6))  # Check shape of output


if __name__ == "__main__":
    unittest.main()
