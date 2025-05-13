import torch
from torch import nn
from .....utils import gradients, hessian


class JastrowKernelElectronNucleiBase(nn.Module):
    def __init__(
        self, nup: int, ndown: int, atomic_pos: torch.Tensor, cuda: bool, **kwargs
    ) -> None:
        r"""Base class for the elec-nuc jastrow factor

        .. math::

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            atoms (torch.tensor): atomic positions of the atoms
            w (float, optional): Value of the variational parameter. Defaults to 1..
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super().__init__()
        self.nup, self.ndown = nup, ndown
        self.cuda = cuda

        self.nelec = nup + ndown
        self.atoms = atomic_pos
        self.natoms = atomic_pos.shape[0]
        self.ndim = 3

        self.device = torch.device("cpu")
        if self.cuda:
            self.device = torch.device("cuda")
        self.requires_autograd = True

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r"""Get the elements of the jastrow matrix :
        .. math::
            out_{i,j} = \exp{ \frac{b r_{i,j}}{1+b'r_{i,j}} }

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the jastrow elements
                          Nbatch x Nelec x Nelec
        """
        raise NotImplementedError()

    def compute_derivative(self, r: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
        """Get the elements of the derivative of the jastrow kernels
        wrt to the first electrons

        .. math::

            d B_{ij} / d k_i =  d B_{ij} / d k_j  = - d B_{ji} / d k_i

            out_{k,i,j} = A1 + A2
            A1_{kij} = w0 \frac{dr_{ij}}{dk_i} / (1 + w r_{ij})
            A2_{kij} = - w0 w' r_{ij} \frac{dr_{ij}}{dk_i} / (1 + w r_{ij})^2

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the derivative of the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """
        if r.requires_grad == False:
            r.requires_grad = True

        with torch.enable_grad():
            kernel = self.forward(r)
            ker_grad = gradients(kernel, r)

        return ker_grad.unsqueeze(1) * dr

    def compute_second_derivative(
        self, r: torch.Tensor, dr: torch.Tensor, d2r: torch.Tensor
    ) -> torch.Tensor:
        """Get the elements of the pure 2nd derivative of the jastrow kernels
        wrt to the first electron

        .. math ::

            d^2 B_{ij} / d k_i^2 =  d^2 B_{ij} / d k_j^2 = d^2 B_{ji} / d k_i^2

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec
            d2r (torch.tensor): matrix of the 2nd derivative of
                                the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the pure 2nd derivative of
                          the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """

        dr2 = dr * dr

        if r.requires_grad == False:
            r.requires_grad = True

        with torch.enable_grad():
            kernel = self.forward(r)
            ker_hess, ker_grad = hessian(kernel, r)
            jhess = (ker_hess).unsqueeze(1) * dr2 + ker_grad.unsqueeze(1) * d2r

        return jhess
