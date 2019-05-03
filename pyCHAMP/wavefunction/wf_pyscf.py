import autograd.numpy as np 
from pyCHAMP.wavefunction.wf_base import WF

from autograd import elementwise_grad as egrad
from autograd import hessian, jacobian
from functools import partial

from pyscf import scf, gto

class PYSCF_WF(WF):

	def __init__(self,atom,basis):

		self.atom = atom
		self.basis = basis

		self.mol = gto.M(atom=self.atom,basis=self.basis)
		self.rhf = scf.RHF(self.mol).run()

		self.config_ground_state = self.rhf.mo_occ.astype('bool')
		self.index_homo = np.max(np.argwhere(self.config_ground_state==True).flatten())

		self.ndim = 3
		self.nelec = np.sum(self.mol.nelec)

	def ao_values(self,pos,func='GTOval_sph'):
		'''Returns the values of all atomic orbitals
		on the position specified by pos.

		Args :
			pos : ndarray shape(N,3)
			func : str Method name
		Returns:
			values : nd array shape(N,Nbasis)
		'''
		return self.mol.eval_gto(func,pos)

	def mo_values(self,pos):
		'''Returns the values of all molecular orbitals
		on the position specified by pos.

		Args :
			pos : ndarray shape(N,3)
		Returns:
			values : nd array shape(N,Nbasis)
		'''
		ao_vals = self.ao_values(pos)
		return np.dot(ao_vals,self.rhf.mo_coeff).T

	def sd_values(self,config,pos):
		'''Returns the values of a given slater determinant
		on the position specified by pos.

		Args :
			config : electronic configuration
			pos : ndarray shape(N,3)
		Returns:
			values : nd array shape(N,Nbasis)
		'''
		mo_vals = self.mo_values(pos)
		return np.linalg.det(mo_vals[np.ix_(config,config)])

	def _excitation(self,i,j):
		'''Create an excitation in c0 frm orb i to orb j.
			orb 0 -> Homo
			orb 1 -> Lumo
			orb -1 -> homo-1
			orb 2 -> lumo+1

			wf._excitation(0,1) excitation from homo to lumo
			wf._excitation(-1,2) excitation from homo-1 to lumo+1
			.....
		'''
		cnew = self.config_ground_state
		cnew[self.index_homo+i] = False
		cnew[self.index_homo+j] = True
		return cnew


	def values(self,param,pos):
		v = []
		for p in pos:
			p = p.reshape(-1,3)
			sp_up = self.sd_values(self.config_ground_state,p[:self.mol.nelec[0],:])
			sp_down = self.sd_values(self.config_ground_state,p[self.mol.nelec[0]:,:])
			v.append(sp_up*sp_down)
		return np.array(v)


	def nuclear_potential(self,pos):
		
		nwalker = pos.shape[0]
		pot = np.zeros(nwalker)
		#pos = pos.T

		for ielec in range(self.nelec):
			epos = pos[:,ielec*self.ndim:(ielec+1)*self.ndim]
			for atpos,atmass in zip(self.mol.atom_coords(),self.mol.atom_mass_list()):
				r = np.sqrt(np.sum((epos-atpos)**2,1))
				pot -= (atmass / r)

		return pot

	def electronic_potential(self,pos):

		nwalker = pos.shape[0]
		pot = np.zeros(nwalker)

		for ielec1 in range(self.nelec-1):
			epos1 = pos[:,ielec1*self.ndim:(ielec1+1)*self.ndim]

			for ielec2 in range(ielec1+1,self.nelec):
				epos2 = pos[:,ielec2*self.ndim:(ielec2+1)*self.ndim]
				r = np.sqrt(np.sum((epos1-epos2)**2,1))

				pot -= (1./r)

		return pot



if __name__ == "__main__":

	wf = PYSCF_WF(atom='O 0 0 0; H 0 0 1; H 0 1 0',basis='dzp')
	nwalkers = 10
	pos = np.random.rand(nwalkers,wf.ndim*wf.nelec)
	v = wf.pdf([],pos)
