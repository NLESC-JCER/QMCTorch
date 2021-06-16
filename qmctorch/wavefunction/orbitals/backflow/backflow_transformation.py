import numpy
import torch
from torch import nn
from ...jastrows.distance.electron_electron_distance import ElectronElectronDistance


class BackFlowTransformation(nn.Module):

    def __init__(self, mol, backflow_kernel, backflow_kernel_kwargs={}, cuda=False):
        """Transform the electorn coordinates into backflow coordinates.
        see : Orbital-dependent backflow wave functions for real-space quantum Monte Carlo
        https://arxiv.org/abs/1910.07167

        .. math:
            \\bold{q}_i = \\bold{r}_i + \\sum_{j\neq i} \\eta(r_{ij})(\\bold{r}_i - \\bold{r}_j)
        """
        super().__init__()
        self.backflow_kernel = backflow_kernel(mol,
                                               cuda,
                                               **backflow_kernel_kwargs)
        self.edist = ElectronElectronDistance(mol.nelec)
        self.nelec = mol.nelec
        self.ndim = 3

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

    def forward(self, pos, derivative=0):

        if derivative == 0:
            return self._backflow(pos)

        elif derivative == 1:
            return self._backflow_derivative(pos)

        elif derivative == 2:
            return self._backflow_second_derivative(pos)

        else:
            raise ValueError(
                'derivative of the backflow transformation must be 0, 1 or 2')

    def _backflow(self, pos):
        """Computes the backflow transformation

        .. math:
            \\bold{q}_i = \\bold{r}_i + \\sum_{j\neq i} \\eta(r_{ij})(\\bold{r}_i - \\bold{r}_j)

        Args:
            pos(torch.tensor): original positions Nbatch x[Nelec*Ndim]

        Returns:
            torch.tensor: transformed positions Nbatch x[Nelec*Ndim]
        """

        # compute the difference
        # Nbatch x Nelec x Nelec x 3
        delta_ee = self.edist.get_difference(
            pos.reshape(-1, self.nelec, self.ndim))

        # compute the backflow function
        # Nbatch x Nelec x Nelec
        bf_kernel = self.backflow_kernel(self.edist(pos))

        # update pos
        pos = pos.reshape(-1, self.nelec, self.ndim) + \
            (bf_kernel.unsqueeze(-1) * delta_ee).sum(2)

        return pos.reshape(-1, self.nelec*self.ndim)

    def _backflow_derivative(self, pos):
        r"""Computes the derivative of the backflow transformation
           wrt the original positions of the electrons

        .. math::
            \\bold{q}_i = \\bold{r}_i + \\sum_{j\\neq i} \\eta(r_{ij})(\\bold{r}_i - \\bold{r}_j)

        .. math::
            \\frac{d q_i}{d x_k} = \\delta_{ik}(1 + \\sum_{j\\neq i} \\frac{d \\eta(r_ij)}{d x_i}(x_i-x_j) + \\eta(r_ij)) +
                                   \\delta_{i\\neq k}(-\\frac{d \\eta(r_ik)}{d x_k}(x_i-x_k) - \\eta(r_ik))

        Args:
            pos(torch.tensor): orginal positions of the electrons Nbatch x[Nelec*Ndim]

        Returns:
            torch.tensor: d q_{i}/d x_k with:
                          q_{i} bf position of elec i
                          x_k original coordinate of the kth elec
                          Nelec x  Nbatch x Nelec x Norb x Ndim
        """

        # ee dist matrix : Nbatch x  Nelec x Nelec
        ree = self.edist(pos)
        nbatch, nelec, _ = ree.shape

        # derivative ee dist matrix : Nbatch x 3 x Nelec x Nelec
        # dr_ij / dx_i = - dr_ij / dx_j
        dree = self.edist(pos, derivative=1)

        # difference between elec pos
        # Nbatch, 3, Nelec, Nelec
        delta_ee = self.edist.get_difference(
            pos.reshape(nbatch, nelec, 3)).permute(0, 3, 1, 2)

        # backflow kernel : Nbatch x 1 x Nelec x Nelec
        bf = self.backflow_kernel(ree)

        # (d eta(r_ij) / d r_ij) (d r_ij/d beta_i)
        # derivative of the back flow kernel : Nbatch x 3 x Nelec x Nelec
        dbf = self.backflow_kernel(ree, derivative=1).unsqueeze(1)
        dbf = dbf * dree

        # (d eta(r_ij) / d beta_i) (alpha_i - alpha_j)
        # Nbatch x 3 x 3 x Nelec x Nelec
        dbf_delta_ee = dbf.unsqueeze(1) * delta_ee.unsqueeze(2)

        # compute the delta_ij * (1 + sum k \neq i eta(rik))
        # Nbatch x Nelec x Nelec (diagonal matrix)
        delta_ij_bf = torch.diag_embed(
            1 + bf.sum(-1), dim1=-1, dim2=-2)

        # eye 3x3 in 1x3x3x1x1
        eye_mat = torch.eye(3, 3).view(1, 3, 3, 1, 1).to(self.device)

        # compute the delta_ab * delta_ij * (1 + sum k \neq i eta(rik))
        # Nbatch x Ndim x Ndim x Nelec x Nelec (diagonal matrix)
        delta_ab_delta_ij_bf = eye_mat * \
            delta_ij_bf.view(nbatch, 1, 1, nelec, nelec)

        # compute sum_k df(r_ik)/dbeta_i (alpha_i - alpha_k)
        # Nbatch x Ndim x Ndim x Nelec x Nelec
        delta_ij_sum = torch.diag_embed(
            dbf_delta_ee.sum(-1), dim1=-1, dim2=-2)

        # compute delta_ab * f(rij)
        delta_ab_bf = eye_mat * bf.view(nbatch, 1, 1, nelec, nelec)

        # return Nbatch x Ndim(alpha) x Ndim(beta) x Nelec(i) x Nelec(j)
        # nbatch d alpha_i / d beta_j
        return delta_ab_delta_ij_bf + delta_ij_sum - dbf_delta_ee - delta_ab_bf

    def _backflow_second_derivative(self, pos):
        r"""Computes the second derivative of the backflow transformation
           wrt the original positions of the electrons

        .. math::
            \\bold{q}_i = \\bold{r}_i + \\sum_{j\\neq i} \\eta(r_{ij})(\\bold{r}_i - \\bold{r}_j)

        .. math::
            \\frac{d q_i}{d x_k} = \\delta_{ik}(1 + \\sum_{j\\neqi} \\frac{d \\eta(r_ij)}{d x_i} + \\eta(r_ij)) +
                                   \\delta_{i\\neq k}(-\\frac{d \\eta(r_ik)}{d x_k} - \\eta(r_ik))

        .. math::
            \\frac{d ^ 2 q_i}{d x_k ^ 2} = \\delta_{ik}(\\sum_{j\\neqi} \\frac{d ^ 2 \\eta(r_ij)}{d x_i ^ 2} + 2 \\frac{d \\eta(r_ij)}{d x_i}) +
                                       - \\delta_{i\\neq k}(\\frac{d ^ 2 \\eta(r_ik)}{d x_k ^ 2} + \\frac{d \\eta(r_ik)}{d x_k})

        Args:
            pos(torch.tensor): orginal positions of the electrons Nbatch x[Nelec*Ndim]

        Returns:
            torch.tensor: d q_{i}/d x_k with:
                          q_{i} bf position of elec i
                          x_k original coordinate of the kth elec
                          Nelec x  Nbatch x Nelec x Norb x Ndim
        """

        # ee dist matrix :
        # Nbatch x  Nelec x Nelec
        ree = self.edist(pos)
        nbatch, nelec, _ = ree.shape

        # difference between elec pos
        # Nbatch, 3, Nelec, Nelec
        delta_ee = self.edist.get_difference(
            pos.reshape(nbatch, nelec, 3)).permute(0, 3, 1, 2)

        # derivative ee dist matrix  d r_{ij} / d x_i
        # Nbatch x 3 x Nelec x Nelec
        dree = self.edist(pos, derivative=1)

        # derivative ee dist matrix :  d2 r_{ij} / d2 x_i
        # Nbatch x 3 x Nelec x Nelec
        d2ree = self.edist(pos, derivative=2)

        # derivative of the back flow kernel : d eta(r_ij)/d r_ij
        # Nbatch x 1 x Nelec x Nelec
        dbf = self.backflow_kernel(ree, derivative=1).unsqueeze(1)

        # second derivative of the back flow kernel : d2 eta(r_ij)/d2 r_ij
        # Nbatch x 1 x Nelec x Nelec
        d2bf = self.backflow_kernel(ree, derivative=2).unsqueeze(1)

        # (d^2 eta(r_ij) / d r_ij^2) (d r_ij/d x_i)^2
        # + (d eta(r_ij) / d r_ij) (d^2 r_ij/d x_i^2)
        # Nbatch x 3 x Nelec x Nelec
        d2bf = (d2bf * dree * dree) + (dbf * d2ree)

        # (d eta(r_ij) / d r_ij) (d r_ij/d x_i)
        # Nbatch x 3 x Nelec x Nelec
        dbf = dbf * dree

        # eye matrix in dim x dim
        eye_mat = torch.eye(3, 3).reshape(
            1, 3, 3, 1, 1).to(self.device)

        # compute delta_ij delta_ab 2 sum_k dbf(ik) / dbeta_i
        term1 = 2 * eye_mat * \
            torch.diag_embed(
                dbf.sum(-1), dim1=-1, dim2=-2).reshape(nbatch, 1, 3, nelec, nelec)

        # (d2 eta(r_ij) / d2 beta_i) (alpha_i - alpha_j)
        # Nbatch x 3 x 3 x Nelec x Nelec
        d2bf_delta_ee = d2bf.unsqueeze(1) * delta_ee.unsqueeze(2)

        # compute sum_k d2f(r_ik)/d2beta_i (alpha_i - alpha_k)
        # Nbatch x Ndim x Ndim x Nelec x Nelec
        term2 = torch.diag_embed(
            d2bf_delta_ee.sum(-1), dim1=-1, dim2=-2)

        # compute delta_ab * df(rij)/dbeta_j
        term3 = 2 * eye_mat * dbf.reshape(nbatch, 1, 3, nelec, nelec)

        # return Nbatch x Ndim(alpha) x Ndim(beta) x Nelec(i) x Nelec(j)
        # nbatch d2 alpha_i / d2 beta_j
        return term1 + term2 + d2bf_delta_ee + term3
