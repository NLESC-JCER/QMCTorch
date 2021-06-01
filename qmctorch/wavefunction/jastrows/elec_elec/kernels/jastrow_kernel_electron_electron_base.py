import torch
from torch import nn
from torch.autograd import grad
from torch import nn


class JastrowKernelElectronElectronBase(nn.Module):

    def __init__(self, nup, ndown, cuda, **kwargs):
        r"""Base class for the elec-elec jastrow kernels

        Args:
            nup ([type]): [description]
            down ([type]): [description]
            cuda (bool, optional): [description]. Defaults to False.
        """

        super().__init__()
        self.nup, self.ndown = nup, ndown
        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

        self.requires_autograd = True

    def forward(self, r):
        r"""Get the elements of the jastrow matrix :


        Args:
            r (torch.tensor): matrix of the e-e distances
                            Nbatch x Nelec_pair

        Returns:
            torch.tensor: matrix fof the jastrow elements
                        Nmo x Nbatch x Nelec_pair

        Note:
            The kernel receives a [Nbatch x Npair] tensor.
            The kernel must first reshape that tensor to a [Nbatch*Npair, 1].
            The kernel must process this tensor to another [Nbatch*Npair, 1] tensor.
            The kenrel must reshape the output to a [Nbatch x Npair] tensor.

        Example:
            >>> def forward(self, x):
            >>>     nbatch, npairs = x.shape
            >>>     x = x.reshape(-1, 1)
            >>>     x = self.fc1(x)
            >>>     ...
            >>>     return(x.reshape(nbatch, npairs))
        """
        raise NotImplementedError()

    def compute_derivative(self, r, dr):
        """Get the elements of the derivative of the jastrow kernels
        wrt to the first electrons using automatic differentiation

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec_pair
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec_pair

        Returns:
            torch.tensor: matrix fof the derivative of the jastrow elements
                          Nmo x Nbatch x Ndim x  Nelec_pair
        """

        if r.requires_grad == False:
            r.requires_grad = True

        with torch.enable_grad():

            kernel = self.forward(r)
            ker_grad = self._grads(kernel, r)

        return ker_grad.unsqueeze(1) * dr

    def compute_second_derivative(self, r, dr, d2r):
        """Get the elements of the pure 2nd derivative of the jastrow kernels
        wrt to the first electron using automatic differentiation

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec_pair
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec_pair
            d2r (torch.tensor): matrix of the 2nd derivative of
                                the e-e distances
                              Nbatch x Ndim x Nelec_pair

        Returns:
            torch.tensor: matrix fof the pure 2nd derivative of
                          the jastrow elements
                          Nmo x Nbatch x Ndim x Nelec_pair
        """
        dr2 = dr * dr

        if r.requires_grad == False:
            r.requires_grad = True

        with torch.enable_grad():

            kernel = self.forward(r)
            ker_hess, ker_grad = self._hess(kernel, r)

            jhess = (ker_hess).unsqueeze(1) * \
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
    def _hess(val, pos):
        """get the hessian of the jastrow values.
        of a given orbital terms
        Warning thos work only because the orbital term are dependent
        of a single rij term, i.e. fij = f(rij)

        Args:
            pos ([type]): [description]
        """

        gval = grad(val,
                    pos,
                    grad_outputs=torch.ones_like(val),
                    create_graph=True)[0]

        hval = grad(gval, pos, grad_outputs=torch.ones_like(gval))[0]

        return hval, gval
