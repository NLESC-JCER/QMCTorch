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


def save_checkpoint(solver, epoch, loss):
    """save the model and optimizer state

    Args:
        solver (solver): solver instance
        epoch (int): epoch
        loss (float): current value of the loss
        filename (str): name to save the file

    Returns:
        float: loss (?)
    """
    filename = 'checkpoint_epoch%d.pth' % epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': solver.wf.state_dict(),
        'optimzier_state_dict': solver.opt.state_dict(),
        'loss': loss
    }, filename)


def load_checkpoint(solver, filename):
    """load a model/optmizer

    Args:
        solver (solver): solver instance
        filename (str): filename

    Returns:
        tuple : solver epoch number and loss
    """
    data = torch.load(filename)
    solver.wf.load_state_dict(data['model_state_dict'])
    solver.opt.load_state_dict(data['optimzier_state_dict'])
    epoch = data['epoch']
    loss = data['loss']
    return solver, epoch, loss


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

    def __init__(self, wf, method='energy', clip=False, penalties=None):
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
        self.use_weight = True
        self.penalties = penalties

        self.loss_fn = {'energy': torch.mean,
                        'variance': torch.var}[method]

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
            local_use_weight = self.use_weight * \
                (not deactivate_weight)
            weight = self.get_sampling_weights(local_use_weight)

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

    def get_clipping_mask(self, local_energies, Nstd=5):
        """computes the clipping mask

        Arguments:
            local_energies {torch.tensor} -- values of the local energies
            Nstd {int} -- Excludes values +/- Nstd x std the mean of the eloc
        """
        if self.clip:
            median = torch.median(local_energies)
            std = torch.std(local_energies)
            emax = median + Nstd * std
            emin = median - Nstd * std
            mask = (
                local_energies < emax) & (
                local_energies > emin)
        else:
            mask = torch.ones_like(
                local_energies).type(torch.bool)

        return mask

    def get_sampling_weights(self, use_weight):
        """Get the weight needed when resampling is not
            done at every step
        """
        if use_weight:

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

    def __init__(self, mat, alpha=0.1):
        """Add a penalty loss to keep the MO orthogonalized

        Keyword Arguments:
            alpha {float} -- strength of the penaly (default: {0.1})
        """
        super(OrthoReg, self).__init__()
        self.W = mat
        self.alpha = alpha

    def forward(self):
        """Return the loss : |W x W^T - I|."""
        return self.alpha * \
            torch.norm(self.W.mm(self.W.transpose(0, 1)) -
                       torch.eye(self.W.shape[0]))
