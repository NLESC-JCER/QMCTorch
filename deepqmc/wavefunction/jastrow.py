import torch
from torch import nn


class ElectronDistance(nn.Module):

    def __init__(self, nelec, ndim):
        super(ElectronDistance, self).__init__()
        self.nelec = nelec
        self.ndim = ndim

    def forward(self, input, derivative=0):
        '''compute the pairwise distance between two sets of electrons.
        Args:
            input1 (Nbatch,Nelec1*Ndim) : position of the electrons
            input2 (Nbatch,Nelec2*Ndim) : position of the electrons
                                          if None -> input1
        Returns:
            mat (Nbatch,Nelec1,Nelec2) : pairwise distance between electrons
        '''

        input = input.view(-1, self.nelec, self.ndim)
        norm = (input**2).sum(-1).unsqueeze(-1)
        dist = norm + norm.transpose(1, 2) - 2.0 * \
            torch.bmm(input, input.transpose(1, 2))

        if derivative == 0:
            return dist

        elif derivative == 1:
            invr = (1./dist).unsqueeze_(1)

            diff_axis = input.transpose(1, 2).unsqueeze_(3)
            diff_axis = diff_axis - diff_axis.transpose(2, 3)

            return diff_axis * invr


class TwoBodyJastrowFactor(nn.Module):

    def __init__(self, nup, ndown):
        super(TwoBodyJastrowFactor, self).__init__()

        self.nup = nup
        self.ndown = ndown
        self.nelec = nup+ndown
        self.ndim = 3

        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.weight.requires_grad = True

        bup = torch.cat((0.25*torch.ones(nup, nup), 0.5 *
                         torch.ones(nup, ndown)), dim=1)

        bdown = torch.cat((0.5*torch.ones(ndown, nup), 0.25 *
                           torch.ones(ndown, ndown)), dim=1)
        self.static_weight = torch.cat((bup, bdown), dim=0)

        self.edist = ElectronDistance(self.nelec, self.ndim)

    def forward(self, pos, derivative=0):
        """Compute the Jastrow factors as :

        .. math::
            J(ri,rj) = \Prod_{i,j} \exp(B_{ij}) with
            B_{ij} = \frac{b r_{i,j}}{1+b'r_{i,j}}

        Args:
            pos ([type]): [description]
            derivative (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """

        r = self.edist(pos)
        jast = self._get_jastrow_elements(r)

        if derivative == 0:
            return self._unique_pair_prod(jast)

        elif derivative == 1:
            return self._jastrow_derivative(r, jast)

    def _get_jastrow_elements(self, r):
        return torch.exp(self.static_weight * r /
                         (1.0 + self.weight * r))

    def _unique_pair_prod(self, mat, not_el=None):

        mat_cpy = mat.clone()
        if not_el is not None:
            i, j = not_el
            mat_cpy[..., i, j] = 1

        return mat_cpy[:, torch.tril(
            torch.ones(self.nelec, self.nelec)) == 0].prod(1).view(-1, 1)

    def _jastrow_derivative(self, r, jast):

        nbatch = r.shape[0]
        dr = self.edist(pos, derivative=1)
        djast = self._get_der_jastrow_elements(r, dr) * jast.unsqueeze(1)

        grad_jast = torch.ones(nbatch, self.nelec, self.ndim)
        for i in range(self.nelec):

            for j in range(i+1, self.nelec):
                grad_jast[:, i] += self._unique_pair_prod(
                    jast, not_el=(i, j)) * djast[..., i, j]

            for j in range(0, i):
                grad_jast[:, i] += self._unique_pair_prod(
                    jast, not_el=(j, i)) * djast[..., j, i]

        return grad_jast

    def _get_der_jastrow_elements(self, r, dr):

        r.unsqueeze_(1)
        a = self.static_weight * dr / (1.0 + self.weight * r)
        b = self.static_weight * self.weight * \
            r / (1.0 + self.weight * r)**2 * dr
        r.squeeze_()
        return (a-b)


class JastrowFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, static_weight):
        '''Compute the Jastrow factor.
        Args:
            input : Nbatch x Nelec x Nelec (elec distance)
            weight : Nelec, Nelec
            static weight : Float
        Returns:
            jastrow : Nbatch x 1
        '''

        # save the tensors
        ctx.save_for_backward(input, weight, static_weight)

        # all jastrow for all electron pairs
        factors = torch.exp(static_weight * input / (1.0 + weight * input))

        # product of the off diag terms
        nr, nc = input.shape[1], input.shape[2]
        factors = factors[:, torch.tril(torch.ones(nr, nc)) == 0].prod(1)

        return factors.view(-1, 1)


if __name__ == "__main__":

    pos = torch.rand(10, 12)
    jastrow = TwoBodyJastrowFactor(2, 2)
    val = jastrow(pos)

    r = jastrow.edist(pos)
    dr = jastrow.edist(pos, derivative=1)

    dval = jastrow(pos, derivative=1)
