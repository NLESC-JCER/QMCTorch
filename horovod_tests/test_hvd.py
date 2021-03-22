import unittest
import horovod.torch as hvd
import torch


class TestHVD(unittest.TestCase):
    def setUp(self):
        hvd.init()

    def test_mpi_support(self):
        assert hvd.mpi_threads_supported()

    def test_horovod_allreduce(self):
        rank = hvd.rank()
        size = hvd.size()

        tensor = torch.zeros(size)
        tensor[rank] = 1
        tensor = hvd.allreduce(tensor, op=hvd.Sum)
        assert (tensor == torch.ones(size)).all()


if __name__ == "__main__":
    t = TestHVD()
    t.setUp()
    t.test_mpi_support()
    t.test_horovod_allreduce()
