import torch
from torch import nn
from torch.autograd import grad, Variable


class BackFlowKernelBase(nn.Module):

    def __init__(self, mol, cuda):
        """Compute the back flow kernel, i.e. the function
        f(rij) where rij is the distance between electron i and j
        This kernel is used in the backflow transformation
        .. math:
            q_i = r_i + \\sum_{j\\neq i} f(r_{ij}) (r_i-r_j)
        """
        super().__init__()
        self.nelec = mol.nelec
        self.cuda = cuda
        self.device = torch.device('cpu')
        if self.cuda:
            self.device = torch.device('cuda')

    def forward(self, ree, derivative=0):
        """Computes the desired values of the kernel
         Args:
            ree (torch.tensor): e-e distance Nbatch x Nelec x Nelec
            derivative (int): derivative requried 0, 1, 2

        Returns:
            torch.tensor : f(r) Nbatch x Nelec x Nelec
        """

        if derivative == 0:
            return self._backflow_kernel(ree)

        elif derivative == 1:
            return self._backflow_kernel_derivative(ree)

        elif derivative == 2:
            return self._backflow_kernel_second_derivative(ree)

        else:
            raise ValueError(
                'derivative of the kernel must be 0, 1 or 2')

    def _backflow_kernel(self, ree):
        """Computes the kernel via autodiff

        Args:
            ree ([type]): [description]

        Returns:
            [type]: [description]
        """
        raise NotImplementedError(
            'Please implement the backflow kernel')

    def _backflow_kernel_derivative(self, ree):
        """Computes the first derivative of the kernel via autodiff

        Args:
            ree ([type]): [description]

        Returns:
            [type]: [description]
        """
        if ree.requires_grad == False:
            ree.requires_grad = True

        with torch.enable_grad():
            kernel_val = self._backflow_kernel(ree)

        return self._grad(kernel_val, ree)

    def _backflow_kernel_second_derivative(self, ree):
        """Computes the second derivative of the kernel via autodiff

        Args:
            ree ([type]): [description]

        Returns:
            [type]: [description]
        """
        if ree.requires_grad == False:
            ree.requires_grad = True

        with torch.enable_grad():

            kernel_val = self._backflow_kernel(ree)
            hess_val, _ = self._hess(kernel_val, ree)

        return hess_val

    @staticmethod
    def _grad(val, ree):
        """Get the gradients of the kernel.

        Args:
            ree ([type]): [description]

        Returns:
            [type]: [description]
        """
        return grad(val, ree, grad_outputs=torch.ones_like(val))[0]

    @staticmethod
    def _hess(val, ree):
        """get the hessian of thekernel.

        Warning thos work only because the kernel term are dependent
        of a single rij term, i.e. fij = f(rij)

        Args:
            pos ([type]): [description]
        """

        gval = grad(val,
                    ree,
                    grad_outputs=torch.ones_like(val),
                    create_graph=True)[0]

        hval = grad(gval, ree, grad_outputs=torch.ones_like(gval))[0]

        return hval, gval
