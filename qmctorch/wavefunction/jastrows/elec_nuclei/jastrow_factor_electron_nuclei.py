import torch
from torch import nn
from typing import Dict, Union, Tuple
from ..distance.electron_nuclei_distance import ElectronNucleiDistance
from ....scf import Molecule
from .kernels.jastrow_kernel_electron_nuclei_base import JastrowKernelElectronNucleiBase


class JastrowFactorElectronNuclei(nn.Module):
    def __init__(
        self,
        mol: Molecule,
        jastrow_kernel: JastrowKernelElectronNucleiBase,
        kernel_kwargs: Dict = {},
        cuda: bool = False,
    ) -> None:
        r"""Base class for two el-nuc jastrow of the form:

        .. math::
            J = \prod_{a,i} \exp(A(r_{ai}))

        Args:
            nup (int): number of spin up electons
            ndow (int): number of spin down electons
            atomic_pos (tensor): positions of the atoms
            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
        """

        super().__init__()

        self.nup = mol.nup
        self.ndown = mol.ndown
        self.nelec = mol.nup + mol.ndown

        self.cuda = cuda
        self.device = torch.device("cpu")
        if self.cuda:
            self.device = torch.device("cuda")

        atomic_pos = torch.as_tensor(mol.atom_coords)
        self.atoms = atomic_pos.to(self.device)
        self.natoms = self.atoms.shape[0]
        self.ndim = 3

        # kernel function
        self.jastrow_kernel = jastrow_kernel(
            mol.nup, mol.ndown, atomic_pos, cuda, **kernel_kwargs
        )

        # requires autograd to compute derivatives
        self.requires_autograd = self.jastrow_kernel.requires_autograd

        # elec-nuc distances
        self.edist = ElectronNucleiDistance(self.nelec, self.atoms, self.ndim)

    def __repr__(self) -> str:
        """representation of the jastrow factor"""
        return "en -> " + self.jastrow_kernel.__class__.__name__

    def forward(
        self,
        pos: torch.Tensor,
        derivative: Union[int, Tuple[int]] = 0,
        sum_grad: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Compute the Jastrow factors.

        Args:
            pos (torch.tensor): Positions of the electrons
                                  Size : Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0,1,2,).
                            Defaults to 0.
            sum_grad (bool, optional): Return the sum_grad (i.e. the sum of
                                       the derivatives) or the individual
                                       terms. Defaults to True.
                                       False only for derivative=1

        Returns:
            torch.tensor: value of the jastrow parameter for all confs
                          derivative = 0  (Nmo) x Nbatch x 1
                          derivative = 1  (Nmo) x Nbatch x Nelec (for sum_grad = True)
                          derivative = 1  (Nmo) x Nbatch x Ndim x Nelec (for sum_grad = False)
                          derivative = 2  (Nmo) x Nbatch x Nelec
        """

        size = pos.shape
        assert size[1] == self.nelec * self.ndim

        r = self.edist(pos)
        kern_vals = self.jastrow_kernel(r)
        jast = torch.exp(kern_vals.sum([-1, -2])).unsqueeze(-1)

        if derivative == 0:
            return jast

        elif derivative == 1:
            dr = self.edist(pos, derivative=1)
            return self.jastrow_factor_derivative(r, dr, jast, sum_grad)

        elif derivative == 2:
            dr = self.edist(pos, derivative=1)
            d2r = self.edist(pos, derivative=2)

            return self.jastrow_factor_second_derivative(r, dr, d2r, jast)

        elif derivative == [0, 1, 2]:
            dr = self.edist(pos, derivative=1)
            d2r = self.edist(pos, derivative=2)

            return (
                jast,
                self.jastrow_factor_derivative(r, dr, jast, sum_grad),
                self.jastrow_factor_second_derivative(r, dr, d2r, jast),
            )

    def jastrow_factor_derivative(
        self, r: torch.Tensor, dr: torch.Tensor, jast: torch.Tensor, sum_grad: bool
    ) -> torch.Tensor:
        """Compute the value of the derivative of the Jastrow factor

        Args:
            r (torch.tensor): ee distance matrix Nbatch x Nelec x Nelec
            jast (torch.tensor): values of the jastrow elements
                                 Nbatch x Nelec x Natom

        Returns:
            torch.tensor: gradient of the jastrow factors
                          Nbatch x Ndim x Nelec
        """
        if sum_grad:
            djast = self.jastrow_kernel.compute_derivative(r, dr).sum((1, 3))
            return djast * jast
        else:
            djast = self.jastrow_kernel.compute_derivative(r, dr).sum(3)
            return djast * jast.unsqueeze(-1)

    def jastrow_factor_second_derivative(
        self, r: torch.Tensor, dr: torch.Tensor, d2r: torch.Tensor, jast: torch.Tensor
    ) -> torch.Tensor:
        """Compute the value of the pure 2nd derivative of the Jastrow factor

        Args:
            r (torch.tensor): ee distance matrix Nbatch x Nelec x Nelec
            jast (torch.tensor): values of the ajstrow elements
                                 Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: diagonal hessian of the jastrow factors
                          Nbatch x Nelec x Ndim
        """
        # pure second derivative terms
        d2jast = self.jastrow_kernel.compute_second_derivative(r, dr, d2r).sum((1, 3))

        # mixed terms
        djast = self.jastrow_kernel.compute_derivative(r, dr)
        djast = ((djast.sum(3)) ** 2).sum(1)

        # add partial derivative
        hess_jast = d2jast + djast

        return hess_jast * jast
