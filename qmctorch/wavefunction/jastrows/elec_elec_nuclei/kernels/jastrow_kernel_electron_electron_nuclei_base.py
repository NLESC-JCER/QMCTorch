import torch
from torch import nn
from torch.autograd import grad
from torch.autograd.variable import Variable


class JastrowKernelElectronElectronNucleiBase(nn.Module):

    def __init__(self, nup, ndown, atomic_pos, cuda, **kwargs):
        r"""Base Class for the elec-elec-nuc jastrow kernel


        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            atoms (torch.tensor): atomic positions of the atoms
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super().__init__()
        self.nup, self.ndown = nup, ndown
        self.cuda = cuda

        self.nelec = nup + ndown
        self.atoms = atomic_pos
        self.natoms = atomic_pos.shape[0]
        self.ndim = 3

        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')
        self.requires_autograd = True

    def forward(self, x):
        """Compute the values of the kernel

        Args:
            x (torch.tensor): e-e and e-n distances distance (Nbatch, Natom, Nelec_pairs, 3)
                              the last dimension holds the values [R_{iA}, R_{jA}, r_{ij}]
                              in that order.

        Returns:
            torch.tensor: values of the kernel (Nbatch, Natom, Nelec_pairs, 1)

        """
        raise NotImplementedError()

    def compute_derivative(self, r, dr):
        """Get the elements of the derivative of the jastrow kernels."""

        kernel = self.forward(r)
        ker_grad = self._grads(kernel, r)

        # return the ker * dr
        out = ker_grad.unsqueeze(1) * dr

        # sum over the atoms
        return out

    def compute_second_derivative(self, r, dr, d2r):
        """Get the elements of the pure 2nd derivative of the jastrow kernels.
        """

        dr2 = dr * dr

        kernel = self.forward(r)
        ker_hess, ker_grad = self._hess(kernel, r, self.device)

        jhess = ker_hess.unsqueeze(1) * \
            dr2 + ker_grad.unsqueeze(1) * d2r

        return jhess

    @staticmethod
    def _grads(val, pos):
        """Get the gradients of the jastrow values
        of a given orbital terms

        Args:
            pos ([type]): [description]

        Returns:
            [type]: [description]
        """
        return grad(val, pos, grad_outputs=torch.ones_like(val))[0]

    @staticmethod
    def _hess(val, pos, device):
        """get the hessian of the jastrow values.

        Args:
            pos ([type]): [description]
        """

        gval = grad(val, pos,
                    grad_outputs=torch.ones_like(val),
                    create_graph=True)[0]

        grad_out = Variable(torch.ones(
            *gval.shape[:-1])).to(device)
        hval = torch.zeros_like(gval).to(device)

        for idim in range(gval.shape[-1]):

            tmp = grad(gval[..., idim], pos,
                       grad_outputs=grad_out,
                       only_inputs=True,
                       create_graph=True)[0]
            hval[..., idim] = tmp[..., idim]

        return hval, gval
