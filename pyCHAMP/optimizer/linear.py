import numpy as np
from pyCHAMP.optimizer.opt_base import OPT_BASE
from scipy.optimize import minimize
import scipy.linalg as spla


class LINEAR(OPT_BASE):

    def __init__(self, wf, maxiter=100, init_param=None, tol=1E-6):
        
        self.wf = wf
        self.tol = tol
        self.maxiter = maxiter
        self.init_param = init_param


    def update_parameters(self,param,pos):
        
        eps = 0.1

        S = self.get_overlap(param,pos)
        H = self.get_hamiltonian(param,pos)
        print(H)
        print(S)

        evals, evects = spla.eig(H,S)
        evals[evals<0] = np.float('Inf')        
        index_min = np.argmin(evals)

        return param + eps*evects[1:,index_min]/evects[0,index_min], False

    def get_overlap(self,param,pos):

        basis = self.get_basis(param,pos)

        n = basis.shape[1]
        S = np.eye(n,n)
        for ib1 in range(n):
            for ib2 in range(n):
                S[ib1,ib2] =  np.dot(basis[:,ib1],basis[:,ib2])

        return S

    def get_hamiltonian(self,param,pos):
        
        HPsi = self.applyHtoBasis(param,pos)
        basis = self.get_basis(param,pos)
        n = basis.shape[1]
        
        H = np.zeros((n,n))

        for ib1 in range(n):
            for ib2 in range(n):
                H[ib1,ib2]  = np.dot(basis[:,ib1],HPsi[:,ib2])
        return H

    def applyHtoBasis(self,param,pos):

        K = self.applyKtoBasis(param,pos)
        V = (self.wf.nuclear_potential(pos) + self.wf.electronic_potential(pos)) * self.wf.values(param,pos)
        return K + V


    def applyKtoBasis(self,param,pos):

        '''Compute the action of the kinetic operator on the we points.
        Args :
            pos : position of the electrons
            metod (string) : mehod to compute the derivatives
        Returns : value of K * psi
        '''

        ndim = pos.shape[1]
        KPsi = np.zeros((pos.shape[0],len(param)+1))
        eps = 1E-6

        for icol in range(ndim):

            pos_tmp = np.copy(pos)
            feps = -2*self.get_basis(param,pos_tmp)

            pos_tmp = np.copy(pos)
            pos_tmp[:,icol] += eps
            feps += self.get_basis(param,pos_tmp)

            pos_tmp = np.copy(pos)
            pos_tmp[:,icol] -= eps
            feps += self.get_basis(param,pos_tmp)

            KPsi += feps/(eps**2)

        return -0.5*KPsi
        
    def get_basis(self,param,pos):
        psi0 = self.wf.values(param,pos)
        psi0 /= np.linalg.norm(psi0)

        psi = self.wf.jacobian_opt(param,pos)

        b = np.hstack((psi0,psi))
        return b

