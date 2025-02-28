from typing import Optional, ContextManager, Tuple
import torch
from torch import nn
from torch.autograd import grad, Variable
from torch.utils.data import Dataset
from math import ceil


def set_torch_double_precision() -> None:
    """Set the default precision to double for all torch tensors."""
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32  = False
    # torch.set_default_tensor_type(torch.DoubleTensor)


def set_torch_single_precision() -> None:
    """Set the default precision to single for all torch tensors."""
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32  = False
    # torch.set_default_tensor_type(torch.FloatTensor)


def fast_power(
    x: torch.Tensor, 
    k: torch.Tensor, 
    mask0: Optional[torch.Tensor] = None, 
    mask2: Optional[torch.Tensor] = None
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


def gradients(
    out: torch.Tensor,
    inp: torch.Tensor,
) -> torch.Tensor:
    """
    Return the gradients of out wrt inp

    Args:
        out (torch.Tensor): The output tensor
        inp (torch.Tensor): The input tensor

    Returns:
        torch.Tensor: Gradient of out wrt inp
    """
    return grad(out, inp, grad_outputs=torch.ones_like(out))


def diagonal_hessian(
        out: torch.Tensor, 
        inp: torch.Tensor, 
        return_grads: bool = False
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
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, inp, grad_outputs=z, only_inputs=True, create_graph=True)[0]

    if return_grads:
        grads = jacob.detach()

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[1]):
        tmp = grad(
            jacob[:, idim], inp, grad_outputs=z, only_inputs=True, create_graph=True
        )[0]

        hess[:, idim] = tmp[:, idim]

    if return_grads:
        return hess, grads

    else:
        return hess


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

    def __getitem__(self, index) -> torch.Tensor:
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


class Loss(nn.Module):
    def __init__(self, 
                 wf, 
                 method: str = "energy", 
                 clip: bool = False):
        """Defines the loss to use during the optimization

        Arguments:
            wf {Wavefunction} -- wave function object used

        Keyword Arguments:
            method {str} -- method to use  (default: {'energy'})
                            (energy, variance, weighted-energy,
                            weighted-variance)
            clip {bool} -- clip the values that are +/- % sigma away from
                           the mean (default: {False})
        """

        super(Loss, self).__init__()

        self.wf = wf
        self.method = method
        self.clip = clip

        # by default we use weights
        # that are needed if we do
        # not resample at every time step
        self.use_weight = True

        # number of +/- std for clipping
        # Excludes values + /- Nstd x std the mean of the eloc
        self.clip_num_std = 5

        # select loss function
        self.loss_fn = {"energy": torch.mean, "variance": torch.var}[method]

        # init values of the weights
        self.weight = {"psi": None, "psi0": None}

    def forward(
        self, 
        pos: torch.Tensor, 
        no_grad: bool = False,
        deactivate_weight: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the loss

        Args:
            pos (torch.Tensor): Positions of the walkers in that batch
            no_grad (bool, optional): Computes the gradient of the loss 
                                      (default: {False})
            deactivate_weight (bool, optional): Deactivates the weight computation 
                                               (default: {False})

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Value of the loss, local energies
        """

        # check if grads are requested
        with self.get_grad_mode(no_grad):
            # compute local eneergies
            local_energies = self.wf.local_energy(pos)

            # mask the energies if necessary
            mask = self.get_clipping_mask(local_energies)

            # sampling_weight
            weight = self.get_sampling_weights(pos, deactivate_weight)

            # compute the loss
            loss = self.loss_fn((weight * local_energies)[mask])

        return loss, local_energies

    @staticmethod
    def get_grad_mode(no_grad: bool) -> ContextManager:
        """Returns a context manager to enable or disable gradient computation.

        Args:
            no_grad (bool): Whether to disable gradient computation.

        Returns:
            typing.ContextManager: A context manager to disable or enable gradient computation.
        """
        return torch.no_grad() if no_grad else torch.enable_grad()

    def get_clipping_mask(self, local_energies: torch.Tensor) -> torch.Tensor:
        """Computes the clipping mask.

        Args:
            local_energies (torch.Tensor): Values of the local energies.

        Returns:
            torch.Tensor: A boolean tensor representing the clipping mask.
        """
        if self.clip:
            median = torch.median(local_energies)
            std = torch.std(local_energies)
            emax = median + self.clip_num_std * std
            emin = median - self.clip_num_std * std
            mask = (local_energies < emax) & (local_energies > emin)
        else:
            mask = torch.ones_like(local_energies).type(torch.bool)

        return mask

    def get_sampling_weights(
        self, pos: torch.Tensor, deactivate_weight: bool
    ) -> torch.Tensor:
        """Get the weight needed when resampling is not
        done at every step

        Args:
            pos (torch.Tensor): Positions of the walkers
            deactivate_weight (bool): Deactivate the computation of the weight

        Returns:
            torch.Tensor: The weight to apply to the local energy
        """

        local_use_weight = self.use_weight * (not deactivate_weight)

        if local_use_weight:
            # computes the weights
            self.weight["psi"] = self.wf(pos)

            # if we just resampled store psi and all w=1
            if self.weight["psi0"] is None:
                self.weight["psi0"] = self.weight["psi"].detach().clone()
                w = torch.ones_like(self.weight["psi"])

            # otherwise compute ration of psi
            else:
                w = (self.weight["psi"] / self.weight["psi0"]) ** 2
                w /= w.sum()  # should we multiply by the number of elements ?

            return w

        else:
            return torch.tensor(1.0)


class OrthoReg(nn.Module):
    """add a penalty to make matrice orthgonal."""

    def __init__(self, alpha: float = 0.1) -> None:
        """Add a penalty loss to keep the MO orthogonalized

        Keyword Arguments:
            alpha {float} -- strength of the penaly (default: {0.1})
        """
        super(OrthoReg, self).__init__()
        self.alpha: float = alpha

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """Return the loss : |W x W^T - I|.

        Args:
            W: The matrix to orthogonalize

        Returns:
            The loss value
        """
        return self.alpha * torch.norm(W.mm(W.transpose(0, 1)) - torch.eye(W.shape[0]))
