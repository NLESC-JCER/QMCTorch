import torch
from torch import nn
from torch.utils.data import Dataset


def set_torch_double_precision():
    """Set the default precision to double for all torch tensors."""
    torch.set_default_dtype = torch.float64
    torch.set_default_tensor_type(torch.DoubleTensor)


def set_torch_single_precision():
    """Set the default precision to single for all torch tensors."""
    torch.set_default_dtype = torch.float32
    torch.set_default_tensor_type(torch.FloatTensor)


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

    def __init__(self, wf, method='variance', clip=False):
        """Defines the loss to use during the optimization

        Arguments:
            wf {WaveFunction} -- wave function object used

        Keyword Arguments:
            method {str} -- method to use  (default: {'variance'})
                            (energy, variance, weighted-energy,
                            weighted-variance)
            clip {bool} -- clip the values that are +/- % sigma away from
                           the mean (default: {False})
        """

        super(Loss, self).__init__()
        self.wf = wf
        self.method = method
        self.clip = clip

        self.use_weight = 'weighted' in self.method
        if self.use_weight:
            self.weight = {'psi': None, 'psi0': None}

    def forward(self, pos, no_grad=False):
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
        _grad = torch.enable_grad()
        if no_grad:
            _grad = torch.no_grad()

        with _grad:

            # compute local eneergies
            local_energies = self.wf.local_energy(pos)

            # mask the energies if necessary
            if self.clip:
                median = torch.median(local_energies)
                std = torch.std(local_energies)
                mask = (local_energies < median +
                        5 * std) & (local_energies > median - 5 * std)
            else:
                mask = torch.ones_like(
                    local_energies).type(torch.bool)

            # un weighted values
            if not self.use_weight:

                if self.method == 'variance':
                    loss = torch.var(local_energies[mask])

                elif self.method == 'energy':
                    loss = torch.mean(local_energies[mask])

                else:
                    raise ValueError(
                        'method must be variance, energy, \
                         weighted-variance or weighted_energy')

            # use weights
            else:

                # computes the weights
                self.weight['psi'] = self.wf(pos)

                if self.weight['psi0'] is None:
                    self.weight['psi0'] = self.weight['psi'].detach(
                    ).clone()

                w = (self.weight['psi'] / self.weight['psi0'])**2
                w /= w.sum()

                if self.method == 'weighted-variance':
                    mu = torch.mean(local_energies)
                    weighted_local_energies = (
                        local_energies - mu)**2 * w

                    # biased variance
                    loss = torch.mean(
                        weighted_local_energies[mask]) * mask.sum()

                    # unbiased variance
                    # loss = weighted_local_energies[mask].sum()/(mask.sum()-1)

                elif self.method == 'weighted-energy':
                    weighted_local_energies = local_energies * w
                    loss = torch.mean(weighted_local_energies[mask])

                else:
                    raise ValueError(
                        'method must be variance, energy, \
                        weighted-variance or weighted_energy')

        return loss, local_energies


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
            torch.norm(W.mm(W.transpose(0, 1)) - torch.eye(W.shape[0]))
