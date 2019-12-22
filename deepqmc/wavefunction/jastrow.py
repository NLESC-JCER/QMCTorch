import torch
from torch import nn


class ElectronDistance(nn.Module):

    def __init__(self, nelec, ndim):
        super(ElectronDistance, self).__init__()
        self.nelec = nelec
        self.ndim = ndim

        _type_ = torch.get_default_dtype()
        if _type_ == torch.float32:
            self.eps = 1E-7
        elif _type_ == torch.float64:
            self.eps = 1E-16

    def forward(self, input, derivative=0):
        """compute the pairwise distance between two sets of electrons
        Or the derivative of these elements

        Args:
            input ([type]): position of the electron Nbatch x [NelecxNdim]
            derivative (int, optional): degre of the derivative.
                                        Defaults to 0.

        Returns:
            torch.tensor: distance (or derivative) matrix
                          Nbatch x Nelec x Nelec if derivative = 0
                          Nbatch x Ndim x  Nelec x Nelec if derivative = 1,2

        """
        '''compute the pairwise distance between two sets of electrons.
        Args:
            input1 (Nbatch,Nelec1*Ndim) : position of the electrons
            input2 (Nbatch,Nelec2*Ndim) : position of the electrons
                                          if None -> input1
        Returns:
            mat (Nbatch,Nelec1,Nelec2) : pairwise distance between electrons
        '''

        input_ = input.view(-1, self.nelec, self.ndim)

        norm = (input_**2).sum(-1).unsqueeze(-1)
        dist = (norm + norm.transpose(1, 2) - 2.0 *
                torch.bmm(input_, input_.transpose(1, 2)))
        eps_ = self.eps * \
            torch.diag(dist.new_ones(dist.shape[-1])).expand_as(dist)
        dist = torch.sqrt(dist + eps_)

        if derivative == 0:
            return dist

        elif derivative == 1:

            eps_ = self.eps * \
                torch.diag(dist.new_ones(dist.shape[-1])).expand_as(dist)

            invr = (1./(dist+eps_)).unsqueeze(1)
            diff_axis = input_.transpose(1, 2).unsqueeze(3)
            diff_axis = diff_axis - diff_axis.transpose(2, 3)
            return diff_axis * invr

        elif derivative == 2:

            eps_ = self.eps * \
                torch.diag(dist.new_ones(dist.shape[-1])).expand_as(dist)
            invr3 = (1./(dist**3+eps_)).unsqueeze(1)
            diff_axis = input_.transpose(1, 2).unsqueeze(3)
            diff_axis = (diff_axis - diff_axis.transpose(2, 3))**2

            diff_axis = diff_axis[:, [[1, 2], [2, 0], [0, 1]], ...].sum(2)
            return (diff_axis * invr3)


class TwoBodyJastrowFactor(nn.Module):

    def __init__(self, nup, ndown, w=1., cuda=False):
        super(TwoBodyJastrowFactor, self).__init__()

        self.nup = nup
        self.ndown = ndown
        self.nelec = nup+ndown
        self.ndim = 3

        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

        self.weight = nn.Parameter(torch.tensor([w]))
        self.weight.requires_grad = True

        bup = torch.cat((0.25*torch.ones(nup, nup), 0.5 *
                         torch.ones(nup, ndown)), dim=1)

        bdown = torch.cat((0.5*torch.ones(ndown, nup), 0.25 *
                           torch.ones(ndown, ndown)), dim=1)

        self.static_weight = torch.cat((bup, bdown), dim=0).to(self.device)

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
        size = pos.shape
        assert size[1] == self.nelec*self.ndim

        r = self.edist(pos)
        jast = self._get_jastrow_elements(r)

        if derivative == 0:
            return self._prod_unique_pairs(jast)

        elif derivative == 1:
            dr = self.edist(pos, derivative=1)
            return self._jastrow_derivative(r, dr, jast)

        elif derivative == 2:
            dr = self.edist(pos, derivative=1)
            d2r = self.edist(pos, derivative=2)
            return self._jastrow_second_derivative(r, dr, d2r, jast)

    def _jastrow_derivative(self, r, dr, jast):
        """Compute the value of the derivative of the Jastrow factor

        Args:
            r (torch.tensor): ee distance matrix Nbatch x Nelec x Nelec
            jast (torch.tensor): values of the ajstrow elements
                                 Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: gradient of the jastrow factors
                          Nbatch x Nelec x Ndim
        """

        djast = self._get_der_jastrow_elements(r, dr).sum(1)
        prod_val = self._prod_unique_pairs(jast)

        return (self._sum_unique_pairs(djast, axis=-1) -
                self._sum_unique_pairs(djast, axis=-2)) * prod_val

    def _jastrow_second_derivative(self, r, dr, d2r, jast):
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
        prod_val = self._prod_unique_pairs(jast)
        d2jast = self._get_second_der_jastrow_elements(
            r, dr, d2r).sum(1)
        hess_jast = (self._sum_unique_pairs(d2jast, axis=-1)
                     + self._sum_unique_pairs(d2jast, axis=-2))

        # mixed terms
        djast = (self._get_der_jastrow_elements(r, dr)).sum(1)
        hess_jast += self._partial_derivative(djast, out_mat=hess_jast)

        return hess_jast * prod_val

    def _get_jastrow_elements(self, r):
        """Get the elements of the jastrow matrix :
        .. math::
            out_{i,j} = \exp{ \frac{b r_{i,j}}{1+b'r_{i,j}} }

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the jastrow elements
                          Nbatch x Nelec x Nelec
        """
        return torch.exp(self._compute_kernel(r))

    def _compute_kernel(self, r):
        return self.static_weight * r / (1.0 + self.weight * r)

    def _get_der_jastrow_elements(self, r, dr):
        """Get the elements of the derivative of the jastrow matrix
        .. math::

            out_{k,i,j} = A + B
            A_{kij} = b \frac{dr_{ij}}{dk_i} / (1+b'r_{ij})
            B_{kij} = - b b' r_{ij} \frac{dr_{ij}}{dk_i} / (1+b'r_{ij})^ 2

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the derivative of the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """

        r_ = r.unsqueeze(1)
        denom = 1. / (1.0 + self.weight * r_)
        a = self.static_weight * dr * denom
        b = - self.static_weight * self.weight * r_ * dr * denom**2

        return (a + b)

    def _get_second_der_jastrow_elements(self, r, dr, d2r):
        """Get the elements of the pure 2nd derivative of the jastrow matrix

        Args:
            r (torch.tensor): matrix of the e-e distances
                              Nbatch x Nelec x Nelec
            dr (torch.tensor): matrix of the derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec
            d2r (torch.tensor): matrix of the 2nd derivative of the e-e distances
                              Nbatch x Ndim x Nelec x Nelec

        Returns:
            torch.tensor: matrix fof the pure 2nd derivative of the jastrow elements
                          Nbatch x Ndim x Nelec x Nelec
        """

        r_ = r.unsqueeze(1)
        denom = 1. / (1.0 + self.weight * r_)
        dr_square = dr**2
        a = self.static_weight * d2r * denom
        b = -2 * self.static_weight * self.weight * dr_square * denom**2
        c = - self.static_weight * self.weight * r_ * d2r * denom**2
        d = 2 * self.static_weight * self.weight**2 * r_ * dr_square * denom**3

        e = self._get_der_jastrow_elements(r, dr)

        return a + b + c + d + e**2

    def _partial_derivative(self, djast, out_mat=None):

        if out_mat is None:
            nbatch = djast.shape[0]
            out_mat = torch.zeros(nbatch, self.nelec)

        for idx in range(self.nelec):

            index_pairs = [(idx, j) for j in range(
                idx+1, self.nelec)] + [(j, idx) for j in range(0, idx)]

            for p1 in range(len(index_pairs)-1):
                i1, j1 = index_pairs[p1]
                for p2 in range(p1+1, len(index_pairs)):
                    i2, j2 = index_pairs[p2]

                    d1 = djast[..., i1, j1] * (-1)**(i1 > j1)
                    d2 = djast[..., i2, j2] * (-1)**(i2 > j2)
                    out_mat[:, idx] += (d1*d2)

        return out_mat

    def _prod_unique_pairs(self, mat, not_el=None):
        """Compute the product of the lower mat elements

        Args:
            mat (torch.tensor): input matrix [..., N x N]
            not_el (tuple(i,j), optional):
                single element(s) to exclude of the product.
                Defaults to None.

        Returns:
            torch.tensor : value of the product
        """

        mat_cpy = mat.clone()
        if not_el is not None:

            if not isinstance(not_el, list):
                not_el = [not_el]

            for _el in not_el:
                i, j = _el
                mat_cpy[..., i, j] = 1

        return mat_cpy[..., torch.tril(
            torch.ones(self.nelec, self.nelec)) == 0].prod(1).view(-1, 1)

    def _sum_unique_pairs(self, mat, axis=None):
        mat_cpy = mat.clone()
        mat_cpy[..., torch.tril(torch.ones(self.nelec, self.nelec)) == 1] = 0

        if axis is None:
            return mat_cpy.sum()
        else:
            return mat_cpy.sum(axis)


if __name__ == "__main__":

    import torch
    from torch.autograd import grad, gradcheck, Variable
    torch.autograd.set_detect_anomaly(True)
    torch.set_default_tensor_type(torch.DoubleTensor)

    def hess(out, pos):
        # compute the jacobian
        z = Variable(torch.ones(out.shape))
        jacob = grad(out, pos,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]

        # compute the diagonal element of the Hessian
        z = Variable(torch.ones(jacob.shape[0]))
        hess = torch.zeros(jacob.shape)

        for idim in range(jacob.shape[1]):

            tmp = grad(jacob[:, idim], pos,
                       grad_outputs=z,
                       only_inputs=True,
                       # create_graph is REQUIRED and
                       # is causing memory issues
                       # for large systems
                       create_graph=True)[0]

            hess[:, idim] = tmp[:, idim]

        return hess

    torch.manual_seed(1)
    pos = torch.rand(4, 12)
    pos.requires_grad = True

    n1, n2 = 2, 2
    n = n1+n2
    jastrow = TwoBodyJastrowFactor(n1, n2)

    r = jastrow.edist(pos)
    dr = jastrow.edist(pos, derivative=1)
    dr_grad = grad(r, pos, grad_outputs=torch.ones_like(r))[0]

    r = jastrow.edist(pos)
    d2r = jastrow.edist(pos, derivative=2)
    d2r_grad = hess(r, pos)
    print(2*d2r.sum(), d2r_grad.sum())

    val = jastrow(pos)
    dval = jastrow(pos, derivative=1)
    dval_grad = grad(val, pos, grad_outputs=torch.ones_like(val))[0]
    print(dval)
    print(dval_grad.view(4, n, 3).sum(2))
    
    val = jastrow(pos)
    d2val_grad = hess(val, pos)
    d2val = jastrow(pos, derivative=2)
    print(d2val.sum(), d2val_grad.sum())
