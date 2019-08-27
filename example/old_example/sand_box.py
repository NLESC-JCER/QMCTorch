import autograd.numpy as np

from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.wavefunction.wf_pyscf import PYSCF_WF

from pyCHAMP.optimizer.minimize import MINIMIZE
from pyCHAMP.optimizer.linear import LINEAR
from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.sampler.pymc3 import PYMC3

from pyCHAMP.solver.vmc import VMC

class HarmOsc1D(WF):

    def __init__(self,nelec,ndim):
        WF.__init__(self, nelec, ndim)

    def values(self,parameters,pos):
        ''' Compute the value of the wave function.

        Args:
            parameters : parameters of th wf
            x: position of the electron

        Returns: values of psi
        '''
    
        beta = parameters[0]
        wf = np.exp(-beta*pos**2).reshape(-1,1)

        return wf
    
    # def jacobian_opt(self,param,pos):
    #   psi = self.values(param,pos)
    #   psi /= np.linalg.norm(psi)
    #   return pos**2*psi

    def nuclear_potential(self,pos):
        return 0.5*pos**2 

    def electronic_potential(self,pos):
        return 0

class HarmOsc2D(WF):

    def __init__(self,nelec,ndim):
        WF.__init__(self, nelec, ndim)

    def values(self,parameters,pos):
        ''' Compute the value of the wave function.

        Args:
            parameters : parameters of th wf
            x: position of the electron

        Returns: values of psi
        '''
    
        b0,b1 = 5,2
        pos = pos.T
        v =   np.exp(- b0 *(pos[0]-1)**2) * np.exp(- b0 *(pos[1])**2)
        v +=  np.exp(- b1 *(pos[0]+1)**2) * np.exp(- b1 *(pos[1])**2) 

        return v
    
    # def jacobian_opt(self,param,pos):
    #   psi = self.values(param,pos)
    #   psi /= np.linalg.norm(psi)
    #   return pos**2*psi

    def nuclear_potential(self,pos):
        return 0.5*pos**2 

    def electronic_potential(self,pos):
        return 0


if __name__ == "__main__":

    #wf = HarmOsc2D(nelec=1, ndim=2)
    wf = PYSCF_WF(atom='O 0 0 0; H 0 0 1; H 0 1 0',basis='dzp')
    #sampler = METROPOLIS(nwalkers=100, nstep=100, step_size = 3, 
    #	                  nelec=wf.nelec, ndim=wf.ndim, domain = {'min':-5,'max':5})
    sampler = PYMC3(nwalkers=100, ndim = wf.nelec*wf.ndim)

    optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)
    optlin = LINEAR(wf=wf,maxiter=25,tol=1E-6)


    # VMS solver
    vmc = VMC(wf=wf, sampler=sampler, optimizer=None)
    pos = vmc.sample([])
    vmc.plot_density(pos)
    e = vmc.wf.energy([],pos)




    # single point
    # opt_param = [0.5]
    # pos,e,s = vmc.single_point(opt_param)
    # print('Energy   : ', e)
    # print('Variance : ', s)
    # vmc.plot_density(pos)

    # optimization
    # init_param = [1.25]
    # vmc.optimize(init_param)
    # vmc.plot_history()
    

    # metro = METROPOLIS(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=1, domain = {'min':-2,'max':2})
    # optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)



    # vmc = VMC(wf=wf, sampler=metro, optimizer=optimizer)
    # pos = vmc.sample(opt_param)


    # diff = DIFFUSION(nwalkers=1000, nstep=1, step_size = 0.5, nelec=1, ndim=1, domain = {'min':-2,'max':2})
    # diff.set_initial_guess(pos)

    # dmc = DMC(wf=wf, sampler=diff, optimizer=None)
    # pos,e,s = dmc.single_point(opt_param)
    # dmc.plot_density(pos)




    # sampler = METROPOLIS(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=1, domain = {'min':-2,'max':2})
    # optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)
    # vmc = VMC(wf=wf, sampler=sampler, optimizer=optimizer)
    # x0 = [1.25]
    # vmc.optimize(x0)

    # plt.plot(vmc.history['energy'])
    # plt.plot(vmc.history['variance'])
    # plt.show()


