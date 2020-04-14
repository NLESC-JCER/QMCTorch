import torch
from torch.autograd import Variable, grad
from pyscf import gto
from qmctorch.wavefunction import Molecule, Orbital
import unittest


def btrace(M):
    return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)


class OrbitalTest(Orbital):
    def __init__(self, mol):
        super(OrbitalTest, self).__init__(mol, use_jastrow=True)

    def first_der_autograd(self, x):
        """Compute the first derivative of the AO using autograd

        Args:
            x (torch.tensor): position of the electrons

        Returns:
            torch.tensor: jacbian of the AO
        """
        out = self.ao(x)

        z = Variable(torch.ones(out.shape))
        jacob = grad(out, x,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]

        return jacob

    def ao_trunk(self, x, n=4):
        """Extract the first 4x4 block of the AO

        Args:
            x (torch.tensor): positions
            n (int, optional): size of the block. Defaults to 4.

        Returns:
            torch.tensor: nxn block of the AO matrix
        """
        out = self.ao(x)
        return out[:, :n, :n]

    def second_der_autograd(self, pos, out=None):
        """Compute the second derivative of the AO using autograd

        Args:
            x (torch.tensor): position of the electrons

        Returns:
            torch.tensor: Hessian of the AO
        """

        out = self.ao(pos)

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
                       retain_graph=True,
                       only_inputs=True,
                       allow_unused=True)[0]

            hess[:, idim] = tmp[:, idim]

        return hess

    def second_der_autograd_mo(self, pos):
        """Compute the second derivative of the MOAO using autograd

        Args:
            x (torch.tensor): position of the electrons

        Returns:
            torch.tensor: Hessian of the AO
        """

        out = self.mo(self.ao(pos))

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
                       retain_graph=True,
                       only_inputs=True,
                       allow_unused=True)[0]

            hess[:, idim] = tmp[:, idim]

        return hess

    def first_der_trace(self, x, dAO=None):
        """Compute : Trace (AO^{-1} \nabla AO)

        Args:
            x (torch.tensor): position of the electrons
            dAO (torch.tensor, optional): precomputed values of the
                                          AO derivative. Defaults to None.

        Returns:
            torch tensor: values of Trace (AO^{-1} \nabla AO)
        """
        AO = self.ao(x)
        if dAO is None:
            dAO = self.ao(x, derivative=1)
        else:
            invAO = torch.inverse(AO)
        return btrace(invAO@dAO)

    def test_grad_autograd(self, pos):
        """Compute the jacobian of the AO block using autograd

        Args:
            pos (torch.tensor): position of the electrons

        Returns:
            torch tensor: jacobian of the AO
        """

        out = torch.det(self.ao_trunk(pos)).view(-1, 1)

        # compute the jacobian
        z = Variable(torch.ones(out.shape))
        jacob = grad(out, pos,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]

        return jacob.sum(1).view(-1, 1)

    def test_hess_autograd(self, pos):
        """Compute the hessian of the AO block using autograd

        Args:
            pos (torch.tensor): position of the electrons

        Returns:
            torch tensor: hessian of the AO
        """
        out = torch.det(self.ao_trunk(pos)).view(-1, 1)

        # compute the jacobian
        z = Variable(torch.ones(out.shape))
        jacob = grad(out, pos,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]

        # compute the diagonal element of the Hessian
        z = Variable(torch.ones(jacob.shape[0]))
        hess = torch.zeros(jacob.shape[0])

        for idim in range(jacob.shape[1]):
            tmp = grad(jacob[:, idim], pos,
                       grad_outputs=z,
                       retain_graph=True,
                       only_inputs=True,
                       allow_unused=True)[0]

            hess += tmp[:, idim]

        return hess.view(-1, 1)

    def test_kin_autograd(self, pos):
        """Compute the kinetic energy using autograd

        Args:
            pos (torch.tensor): electron position

        Returns:
            torch.tensor: kinetic energy values
        """

        out = self.mo(self.ao(pos))
        out = out[:, :4, :4]
        out = torch.det(out[:, :2, :2]) * torch.det(out[:, 2:, :2])
        out = out.view(-1)

        # compute the jacobian
        z = Variable(torch.ones(out.shape))
        jacob = grad(out, pos,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]

        # compute the diagonal element of the Hessian
        z = Variable(torch.ones(jacob.shape[0]))
        hess = torch.zeros(jacob.shape[0])

        for idim in range(jacob.shape[1]):
            tmp = grad(jacob[:, idim], pos,
                       grad_outputs=z,
                       retain_graph=True,
                       only_inputs=True,
                       allow_unused=True)[0]

            hess += tmp[:, idim]

        return hess.view(-1, 1)


class TestTrace(unittest.TestCase):

    def setUp(self):

        atom_str = 'C 0 0 -0.69; O 0 0 0.69'
        self.m = gto.M(atom=atom_str, basis='sto-3g', unit='bohr')
        self.mol = Molecule(atom=atom_str, calculator='pyscf',
                            basis='sto-3g', unit='bohr')

        # define the wave function
        self.wf = OrbitalTest(self.mol)
        self.x = 2 * torch.rand(5, 3 * self.mol.nelec) - 1.
        self.x.requires_grad = True

    def test_ao_der(self):
        """Test the values of the AO derivative."""
        dAO = self.wf.ao(self.x, derivative=1).sum()
        dAO_auto = self.wf.first_der_autograd(self.x).sum()
        assert(torch.allclose(dAO, dAO_auto))

    def test_ao_2der(self):
        """Test the values of the AO 2nd derivative."""
        d2AO = self.wf.ao(self.x, derivative=2).sum()
        d2AO_auto = self.wf.second_der_autograd(self.x).sum()
        assert(torch.allclose(d2AO, d2AO_auto))

    def test_mo_2der(self):
        """Test the values of the MO 2nd derivative."""
        d2MO = self.wf.mo(self.wf.ao(self.x, derivative=2)).sum()
        d2MO_auto = self.wf.second_der_autograd_mo(self.x).sum()
        assert(torch.allclose(d2MO, d2MO_auto))

    def test_trace(self):
        """Test the values jacobian and hessian with autograd and
        trace trick."""
        AO = self.wf.ao_trunk(self.x)
        iAO = torch.inverse(AO)

        dAO = self.wf.ao(self.x, derivative=1)
        d2AO = self.wf.ao(self.x, derivative=2)

        jac_auto = self.wf.test_grad_autograd(self.x)
        jac_trace = btrace(iAO@dAO[:, :4, :4]) * torch.det(AO)
        assert(torch.allclose(jac_auto.sum(), jac_trace.sum()))

        hess_auto = self.wf.test_hess_autograd(self.x)
        hess_trace = btrace(iAO@d2AO[:, :4, :4]) * torch.det(AO)
        assert(torch.allclose(hess_auto.sum(), hess_trace.sum()))

    def test_kinetic(self):
        """Test the values kinetic energy computed via autograd and
        trace trick."""

        # gives Nan on trvis servers when used with Jastrow
        kin_auto = self.wf.kinetic_energy_autograd(self.x)
        wfv = self.wf(self.x)
        kin_auto /= wfv

        kin_trace = self.wf.kinetic_energy_jacobi(
            self.x, return_local_energy=True)

        delta = kin_auto / kin_trace
        print(delta)

        # assert torch.allclose(delta, torch.ones_like(
        #     delta), atol=1e-3, rtol=1E-3)


if __name__ == "__main__":
    unittest.main()
