import unittest
import horovod.torch as hvd


class TestHVD(unittest.TestCase):
    def test_mpi_support(self):
        hvd.init()
        assert (hvd.mpi_threads_supported())


if __name__ == "__main__":
    t = TestHVD()
    t.test_mpi_support()
