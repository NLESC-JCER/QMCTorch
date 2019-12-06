
import torch
from torch.autograd import Variable, grad
from pyscf import gto
from deepqmc.wavefunction.wf_orbital import Orbital

from deepqmc.wavefunction.molecule import Molecule


# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree

def btrace(M):
    return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)


class OrbitalH2(Orbital):
    def __init__(self, mol):
        super(OrbitalH2, self).__init__(mol)

    # def pool(self,x):
    #     return (x[:,0,0]*x[:,1,0]).view(-1,1)

    def first_der_autograd(self, x):

        out = self.ao(x)

        z = Variable(torch.ones(out.shape))
        jacob = grad(out, x,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]

        return jacob

    def ao_trunk(self, x, n=4):
        out = self.ao(x)
        return out[:, :n, :n]

    def second_der_autograd(self, pos, out=None):

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

    def second_der_autograd_mo(self, pos, out=None):

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

        AO = self.ao(x)
        if dAO is None:
            dAO = self.ao(x, derivative=1)
        else:
            invAO = torch.inverse(AO)
        return btrace(invAO@dAO)

    def test_grad_autograd(self, pos, out=None):

        out = torch.det(self.ao_trunk(pos)).view(-1, 1)

        # compute the jacobian
        z = Variable(torch.ones(out.shape))
        jacob = grad(out, pos,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]

        return jacob.sum(1).view(-1, 1)

    def test_hess_autograd(self, pos, out=None):

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

    def test_kin_autograd(self, pos, out=None):

        out = self.mo(self.ao(pos))
        out = out[:, :4, :4]
        out = torch.det(out[:, :2, :2])*torch.det(out[:, 2:, :2])
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


# define the molecule
# mol = Molecule(atom='H 0 0 -0.37; H 0 0 0.37', basis_type='sto', basis='sz')

atom_str = 'C 0 0 -0.69; O 0 0 0.69'
m = gto.M(atom=atom_str, basis='sto-3g', unit='bohr')
mol = Molecule(atom=atom_str, basis_type='gto', basis='sto-3g', unit='bohr')

# define the wave function
wf = OrbitalH2(mol)

x = Variable(torch.rand(5, 3*mol.nelec))
# x = Variable(torch.ones(3,3*mol.nelec))
x.requires_grad = True


print('\n-- Comparison of the AO derivatives')
dAO = wf.ao(x, derivative=1)
dAO_auto = wf.first_der_autograd(x)
print(dAO.sum())
print(dAO_auto.sum())


print('\n-- Comparison of the AO 2nd derivatives')
d2AO = wf.ao(x, derivative=2)
d2AO_auto = wf.second_der_autograd(x)
print(d2AO.sum())
print(d2AO_auto.sum())


print('\n-- Comparison of the MO 2nd derivatives')
d2MO = wf.mo(wf.ao(x, derivative=2))
d2MO_auto = wf.second_der_autograd_mo(x)
print(d2MO.sum())
print(d2MO_auto.sum())


AO = wf.ao_trunk(x)
iAO = torch.inverse(AO)


print('\n-- Comparison of the Jacobian of det(AO)')
jac_auto = wf.test_grad_autograd(x)
jac_trace = btrace(iAO@dAO[:, :4, :4])*torch.det(AO)
print(jac_auto.sum())
print(jac_trace.sum())


print('\n-- Comparison of the Hessian of det(AO)')
hess_auto = wf.test_hess_autograd(x)
hess_trace = btrace(iAO@d2AO[:, :4, :4])*torch.det(AO)
print(hess_auto.sum())
print(hess_trace.sum())


print('\n-- Comparison of the test kinetic energy')
test_kin_auto = wf.test_kin_autograd(x)
MO = wf.mo(wf.ao(x))
MO = MO[:, :4, :4]

MO1 = MO[:, :2, :2]
MO2 = MO[:, 2:, :2]
iMO1 = torch.inverse(MO1)
iMO2 = torch.inverse(MO2)

d2MO = d2AO[:, :4, :4]
d2MO1 = d2MO[:, :2, :2]
d2MO2 = d2MO[:, 2:, :2]

test_kin_trace = (btrace(iMO1@d2MO1) + btrace(iMO2@d2MO2)) * \
    torch.det(MO1)*torch.det(MO2)

print(test_kin_auto.view(-1))
print(test_kin_trace.view(-1))

print('\n-- Comparison of the kinetic energy')
kin_auto = wf.kinetic_energy_autograd(x)
kin_trace = wf.kinetic_energy_jacobi(x)
print(kin_auto.view(-1))
print(kin_trace.view(-1))


# plot the molecule
# plot_molecule(solver)

# optimize the geometry
# solver.configure(task='geo_opt')
# solver.observable(['local_energy','atomic_distances'])
# solver.run(5,loss='energy')

# plot the data
# plot_observable(solver.obs_dict,e0=-1.16)
