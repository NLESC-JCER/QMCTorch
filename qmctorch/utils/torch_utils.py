import torch
from torch import nn
from torch.autograd import grad, Variable
from torch.utils.data import Dataset


def set_torch_double_precision():
    """Set the default precision to double for all torch tensors."""
    torch.set_default_dtype = torch.float64
    torch.set_default_tensor_type(torch.DoubleTensor)


def set_torch_single_precision():
    """Set the default precision to single for all torch tensors."""
    torch.set_default_dtype = torch.float32
    torch.set_default_tensor_type(torch.FloatTensor)


def fast_power(x, k, mask0=None, mask2=None):
    """Computes x**k when k have elements 0, 1, 2

    Args:
        x (torch.tensor): input
        k (torch.tensor): exponents
        mask0 (torch.tensor): precomputed mask of the elements of that are 0 (Defaults to None and computed here)
        mask2 (torch.tensor): precomputed mask of the elements of that are 2 (Defaults to None and computed here)

    Returns:
        torch.tensor: values of x**k
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


def gradients(out, inp):
    """Return the gradients of out wrt inp

    Args:
        out ([type]): [description]
        inp ([type]): [description]
    """
    return grad(out, inp, grad_outputs=torch.ones_like(out))


def diagonal_hessian(out, inp, return_grads=False):
    """return the diagonal hessian of out wrt to inp

    Args:
        out ([type]): [description]
        inp ([type]): [description]

    Returns:
        [type]: [description]
    """
    # compute the jacobian
    z = Variable(torch.ones(out.shape))
    jacob = grad(out, inp,
                 grad_outputs=z,
                 only_inputs=True,
                 create_graph=True)[0]

    if return_grads:
        grads = jacob.detach()

    # compute the diagonal element of the Hessian
    z = Variable(torch.ones(jacob.shape[0]))
    hess = torch.zeros(jacob.shape)

    for idim in range(jacob.shape[1]):

        tmp = grad(jacob[:, idim], inp,
                   grad_outputs=z,
                   only_inputs=True,
                   create_graph=True)[0]

        hess[:, idim] = tmp[:, idim]

    if return_grads:
        return hess, grads

    else:
        return hess


class DataSet(Dataset):

    def __init__(self, data):
        """Creates a torch data set

        Arguments:
            data {torch.tensor} -- data
        """
        self.data = data

    def __len__(self):
        """get the number of data points

        Returns:
            int -- number of data points
        """
        return self.data.shape[0]

    def __getitem__(self, index):
        """returns a given data point

        Arguments:
            index {int} -- index of the point

        Returns:
            torch.tensor -- data of that point
        """
        return self.data[index, :]


class Loss(nn.Module):

    def __init__(
            self,
            wf,
            method='energy',
            clip=False):
        """Defines the loss to use during the optimization

        Arguments:
            wf {WaveFunction} -- wave function object used

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
        self.loss_fn = {'energy': torch.mean,
                        'variance': torch.var}[method]

        # init values of the weights
        self.weight = {'psi': None, 'psi0': None}

    def forward(self, pos, no_grad=False, deactivate_weight=False):
        """Computes the loss

        Arguments:
            pos {torch.tensor} -- positions of the walkers in that batch

        Keyword Arguments:
            no_grad {bool} -- computes the gradient of the loss
                              (default: {False})

        Returns:
            torch.tensor, torch.tensor -- value of the loss, local energies
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
    def get_grad_mode(no_grad):
        """Returns enable_grad or no_grad

        Arguments:
            no_grad {bool} -- [description]
        """

        return torch.no_grad() if no_grad else torch.enable_grad()

    def get_clipping_mask(self, local_energies):
        """computes the clipping mask

        Arguments:
            local_energies {torch.tensor} -- values of the local energies
        """
        if self.clip:
            median = torch.median(local_energies)
            std = torch.std(local_energies)
            emax = median + self.clip_num_std * std
            emin = median - self.clip_num_std * std
            mask = (
                local_energies < emax) & (
                local_energies > emin)
        else:
            mask = torch.ones_like(
                local_energies).type(torch.bool)

        return mask

    def get_sampling_weights(self, pos, deactivate_weight):
        """Get the weight needed when resampling is not
            done at every step
        """

        local_use_weight = self.use_weight * \
            (not deactivate_weight)

        if local_use_weight:

            # computes the weights
            self.weight['psi'] = self.wf(pos)

            # if we just resampled store psi and all w=1
            if self.weight['psi0'] is None:
                self.weight['psi0'] = self.weight['psi'].detach(
                ).clone()
                w = torch.ones_like(self.weight['psi'])

            # otherwise compute ration of psi
            else:
                w = (self.weight['psi'] / self.weight['psi0'])**2
                w /= w.sum()  # should we multiply by the number of elements ?

            return w

        else:
            return 1.


class OrthoReg(nn.Module):
    '''add a penalty to make matrice orthgonal.'''

    def __init__(self, alpha=0.1):
        """Add a penalty loss to keep the MO orthogonalized

        Keyword Arguments:
            alpha {float} -- strength of the penaly (default: {0.1})
        """
        super(OrthoReg, self).__init__()
        self.alpha = alpha

    def forward(self, W):
        """Return the loss : |W x W^T - I|."""
        return self.alpha * \
            torch.norm(W.mm(W.transpose(0, 1)) -
                       torch.eye(W.shape[0]))
