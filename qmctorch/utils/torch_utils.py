from typing import Optional, Tuple
import torch
from torch.autograd import grad, Variable
from torch.utils.data import Dataset
from math import ceil


def set_torch_double_precision() -> None:
    """Set the default precision to double for all torch tensors."""
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # torch.set_default_tensor_type(torch.DoubleTensor)


def set_torch_single_precision() -> None:
    """Set the default precision to single for all torch tensors."""
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # torch.set_default_tensor_type(torch.FloatTensor)


def fast_power(
    x: torch.Tensor,
    k: torch.Tensor,
    mask0: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes x**k when k have elements 0, 1, 2.

    Args:
        x (torch.Tensor): input
        k (torch.Tensor): exponents
        mask0 (torch.Tensor): precomputed mask of the elements of that are 0 (Defaults to None and computed here)
        mask2 (torch.Tensor): precomputed mask of the elements of that are 2 (Defaults to None and computed here)

    Returns:
        torch.Tensor: values of x**k
    """
    kmax = 3
    if k.max() < kmax:
        out = x.clone()

        if mask0 is None:
            mask0 = k == 0

        out.masked_fill_(mask0, 1)

        if k.max() > 1:
            if mask2 is None:
                mask2 = k == 2
            out[..., mask2] *= out[..., mask2]

    else:
        out = x**k

    return out


def gradients(out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """
    Return the gradients of out wrt inp

    Args:
        out (torch.Tensor): The output tensor
        inp (torch.Tensor): The input tensor

    Returns:
        torch.Tensor: Gradient of out wrt inp
    """
    gval = grad(out, inp, grad_outputs=torch.ones_like(out))[0]
    return gval.detach()


def hessian(out: torch.Tensor, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the Hessian and the gradient of `out` wrt `inp`.

    Args:
        out (torch.Tensor): The output tensor.
        inp (torch.Tensor): The input tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The Hessian and the gradient.
    """
    gval = grad(out, inp, grad_outputs=torch.ones_like(out), create_graph=True)[0]
    hval = grad(gval, inp, grad_outputs=torch.ones_like(gval))[0]

    return hval, gval.detach()


def diagonal_hessian(
    out: torch.Tensor,
    inp: torch.Tensor,
) -> torch.Tensor:
    """Return the diagonal Hessian of `out` with respect to `inp`.

    Args:
        out (torch.Tensor): The output tensor.
        inp (torch.Tensor): The input tensor.
        return_grads (bool, optional): Whether to return gradients. Defaults to False.

    Returns:
        torch.Tensor: Diagonal elements of the Hessian.
        torch.Tensor (optional): Gradients of `out` with respect to `inp` if `return_grads` is True.
    """
    # compute the jacobian
    jacob = grad(
        out, inp, grad_outputs=torch.ones_like(out), only_inputs=True, create_graph=True
    )[0]

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros_like(jacob)

    for idim in range(jacob.shape[1]):
        tmp = grad(
            jacob[:, idim], inp, grad_outputs=z, only_inputs=True, create_graph=True
        )[0]

        hess[:, idim] = tmp[:, idim]

    return hess, jacob.detach()


class DataSet(Dataset):
    def __init__(self, data: torch.Tensor) -> None:
        """Creates a torch data set

        Arguments:
            data (torch.Tensor): data
        """

    def __len__(self) -> int:
        """get the number of data points

        Returns:
            int -- number of data points
        """
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """returns a given data point

        Arguments:
            index {int} -- index of the point

        Returns:
            torch.tensor -- data of that point
        """
        return self.data[index, :]


class DataLoader:
    def __init__(
        self, data: torch.Tensor, batch_size: int, pin_memory: bool = False
    ) -> None:
        """Simple DataLoader to replace torch data loader

        Args:
            data (torch.Tensor): data to load [Nbatch,Nelec*3]
            batch_size (int): size of the minibatch
            pin_memory (bool, optional): copy the data to pinned memory. Defaults to False.
        """

        if pin_memory:
            self.dataset = data.pin_memory()
        else:
            self.dataset = data

        self.len = len(data)
        self.nbatch = ceil(self.len / batch_size)
        self.count = 0
        self.batch_size = batch_size

    def __iter__(self):
        """Initialize the iterator.

        Returns:
            DataLoader: The iterator instance.
        """
        self.count = 0
        return self

    def __next__(self) -> torch.Tensor:
        """Returns the next batch of data points.

        Returns:
            torch.Tensor: The next batch of data points.

        Raises:
            StopIteration: If there are no more batches to return.
        """
        if self.count < self.nbatch - 1:
            out = self.dataset[
                self.count * self.batch_size : (self.count + 1) * self.batch_size
            ]
            self.count += 1
            return out
        elif self.count == self.nbatch - 1:
            out = self.dataset[self.count * self.batch_size :]
            self.count += 1
            return out
        else:
            raise StopIteration
