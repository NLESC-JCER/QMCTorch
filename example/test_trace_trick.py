import sys
import torch
from torch.autograd import Variable, grad
from torch.optim import Adam
from pyscf import gto
from deepqmc.wavefunction.wf_orbital import Orbital
#from deepqmc.solver.solver_orbital_distributed import DistSolverOrbital  as SolverOrbital
import numpy as np

from deepqmc.wavefunction.molecule import Molecule


# bond distance : 0.74 A -> 1.38 a
# optimal H positions +0.69 and -0.69
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree

def btrace(M):
    return torch.diagonal(M,dim1=-2,dim2=-1).sum(-1)

class OrbitalH2(Orbital):
    def __init__(self,mol):
        super(OrbitalH2,self).__init__(mol)

    def pool(self,x):
        return (x[:,0,0]*x[:,1,0]).view(-1,1)


    def first_der_autograd(self,x):

        out = self.ao(x)        
        #out = torch.det(out)


        z = Variable(torch.ones(out.shape))
        jacob = grad(out,x,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]

        return jacob

    def second_der_autograd(self,pos,out=None):


        out = self.ao(pos)

        # compute the jacobian            
        z = Variable(torch.ones(out.shape))
        jacob = grad(out,pos,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]
        
        # compute the diagonal element of the Hessian
        z = Variable(torch.ones(jacob.shape[0]))
        hess = torch.zeros(jacob.shape)
        
        for idim in range(jacob.shape[1]):
            tmp = grad(jacob[:,idim],pos,
                      grad_outputs=z,
                      retain_graph=True,
                      only_inputs=True,
                      allow_unused=True)[0]    
            
            hess[:,idim] = tmp[:,idim]
        
        return hess

    def first_der_trace(self,x,dAO = None):

        AO = self.ao(x)
        if dAO is None:
            dAO = self.ao(x,derivative=1)
        else:
            invAO = torch.inverse(AO)
        return btrace(invAO@dAO)


    def test_grad_autograd(self,pos,out=None):
        
        out = torch.det(self.ao(pos)).view(-1,1)

        # compute the jacobian            
        z = Variable(torch.ones(out.shape))
        jacob = grad(out,pos,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]

        return jacob.sum(1).view(-1,1) / out

    def test_kin_autograd(self,pos,out=None):
        
        out = torch.det(self.ao(pos)).view(-1,1)

        # compute the jacobian            
        z = Variable(torch.ones(out.shape))
        jacob = grad(out,pos,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]
        
        # compute the diagonal element of the Hessian
        z = Variable(torch.ones(jacob.shape[0]))
        hess = torch.zeros(jacob.shape[0])
        
        for idim in range(jacob.shape[1]):
            tmp = grad(jacob[:,idim],pos,
                      grad_outputs=z,
                      retain_graph=True,
                      only_inputs=True,
                      allow_unused=True)[0]    
            
            hess += tmp[:,idim]
        
        return  hess.view(-1,1) / out


# define the molecule
#mol = Molecule(atom='H 0 0 -0.37; H 0 0 0.37', basis_type='sto', basis='sz')

atom_str = 'Li 0 0 -0.69; H 0 0 0.69'
m = gto.M(atom=atom_str, basis='sto-3g',unit='bohr')
mol = Molecule(atom=atom_str, basis_type='gto', basis='sto-3g',unit='bohr')

# define the wave function
wf = OrbitalH2(mol)

x = Variable(torch.rand(5,3*mol.nelec))
#x = Variable(torch.ones(3,3*mol.nelec))
x.requires_grad=True

dAO  = wf.ao(x,derivative=1)
dAO_auto = wf.first_der_autograd(x)

print(dAO.sum())
print(dAO_auto.sum())

d2AO  = wf.ao(x,derivative=2)
d2AO_auto = wf.second_der_autograd(x)

print(d2AO.sum())
print(d2AO_auto.sum())

# AO = wf.ao(x)
# pos = x.detach().numpy().reshape(5,mol.nelec,3)
# AOref = []
# for i in range(5):
#     AOref.append(m.eval_gto('GTOval_cart',pos[i]))
# AOref = np.array(AOref).reshape(5,mol.nelec,mol.norb)


# iA = torch.inverse(A)


# jac_auto = wf.test_grad_autograd(x) 
# jac_trace = btrace(iA@dAO)

# hess_auto = wf.test_kin_autograd(x)
# hess_trace = btrace(iA@d2AO)


# MO = wf.mo(A)
# d2MO = wf.mo(d2AO)
# kin_auto = wf.kinetic_energy(x)
# kin_trace = wf.kinpool(MO,d2MO)


# plot the molecule
#plot_molecule(solver)

# optimize the geometry
#solver.configure(task='geo_opt')
#solver.observable(['local_energy','atomic_distances'])
#solver.run(5,loss='energy')

# plot the data
#plot_observable(solver.obs_dict,e0=-1.16)











