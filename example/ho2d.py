import autograd.numpy as np

from pyCHAMP.wavefunction.wf_base import WF

from pyCHAMP.optimizer.minimize import MINIMIZE
#from pyCHAMP.optimizer.swarm import SWARM

from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.sampler.pymc3 import PYMC3
#from pyCHAMP.sampler.hamiltonian import HAMILTONIAN
from pyCHAMP.solver.vmc import VMC





class HarmOsc2D(WF):

    def __init__(self,nelec=1,ndim=2):
        WF.__init__(self, nelec, ndim)

    def values(self,param,pos):

        b0 = param[0]
        pos = pos.T
        v =   np.exp(- b0 *(pos[0])**2) * np.exp(- b0 *(pos[1])**2)
        return v

    def nuclear_potential(self,pos):
        return np.sum(0.5*pos**2,1)

    def electronic_potential(self,pos):
        return 0


if __name__ == "__main__":

    wf = HarmOsc2D(nelec=1, ndim=2)
    sampler = METROPOLIS(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=2, domain = {'min':-2,'max':2})
    #sampler = PYMC3(nwalkers=100,ndim=2)

    #sampler = HAMILTONIAN(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=1)
    optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)

    #optimizer = SWARM( maxiter=25)
    
    # VMC solver
    vmc = VMC(wf=wf, sampler=sampler, optimizer=optimizer)

    # single point
    opt_param = [0.5]   
    pos,e,s = vmc.single_point(opt_param)
    print('Energy   : ', e)
    print('Variance : ', s)
    vmc.plot_density(pos)

    # optimization
    # init_param = [1.]
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


